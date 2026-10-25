"""StudyResult aggregation, reference condition selection, Pareto front, and plot/table specs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.config import RESULT_SCHEMA_VERSION, STUDY_VERSION
from ..core.contracts import DatasetContext, ExperimentResult, StudyResult
from .diagnostics import GRAPH_VIS_CONFIG, LEARNING_CURVE_CONFIG


def build_dataset_profile(dataset: DatasetContext) -> dict[str, pd.DataFrame]:
    """Assemble descriptive tables of class balance, label cardinality, and co-occurrence."""
    y = np.asarray(dataset.targets, dtype=np.float32)
    classes = list(dataset.classes)
    n, c = y.shape
    support = y.sum(axis=0).astype(int)
    prevalence = support / max(n, 1)
    max_support = max(int(support.max()), 1)

    class_balance = (
        pd.DataFrame({
            "class_name": classes,
            "support": support,
            "prevalence": prevalence,
            "imbalance_ratio_vs_max": max_support / np.maximum(support, 1),
            "neg_pos_ratio": (n - support) / np.maximum(support, 1),
        })
        .sort_values("support", ascending=False)
        .reset_index(drop=True)
    )

    cardinality = y.sum(axis=1).astype(int)
    card_dist = (
        pd.Series(cardinality, name="labels_per_item")
        .value_counts()
        .sort_index()
        .rename_axis("labels_per_item")
        .reset_index(name="n_items")
    )

    labelsets = pd.Series([tuple(np.flatnonzero(row)) for row in y])
    cooc = y.T @ y
    union = support[:, None] + support[None, :] - cooc
    jaccard = np.divide(cooc, union, out=np.zeros_like(cooc, dtype=float), where=union > 0)
    cooc_count = pd.DataFrame(cooc.astype(int), index=classes, columns=classes)
    cooc_jaccard = pd.DataFrame(jaccard, index=classes, columns=classes)

    fold_rows = []
    for f in sorted(np.unique(dataset.fold_id)):
        s = y[dataset.fold_id == f].sum(axis=0).astype(int)
        fold_rows.extend(
            {"fold": int(f) + 1, "class_name": cl, "support": int(v)}
            for cl, v in zip(classes, s)
        )
    fold_support = pd.DataFrame(fold_rows)
    fold_matrix = fold_support.pivot(index="class_name", columns="fold", values="support").reindex(
        classes
    )

    overview = pd.DataFrame([{
        "n_items": n,
        "n_classes": c,
        "mean_label_cardinality": float(cardinality.mean()),
        "multilabel_density": float(cardinality.mean() / c),
        "unique_labelsets": int(labelsets.nunique()),
        "support_min": int(support.min()),
        "support_max": int(support.max()),
        "max_min_support_ratio": float(support.max() / max(int(support.min()), 1)),
    }])
    return {
        "dataset_overview": overview,
        "dataset_class_balance": class_balance,
        "dataset_label_cardinality": card_dist,
        "dataset_cooccurrence_count": cooc_count,
        "dataset_cooccurrence_jaccard": cooc_jaccard,
        "dataset_fold_support": fold_support,
        "dataset_fold_support_matrix": fold_matrix,
    }


def summarize_experiment(
    experiment_result: ExperimentResult,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute fold-averaged metric and timing summaries across experimental conditions."""
    cond = experiment_result.conditions
    obs = experiment_result.metric_observations
    by_fold = obs.groupby(["condition_id", "metric", "fold"], as_index=False).value.mean()
    summary_long = by_fold.groupby(["condition_id", "metric"], as_index=False).agg(
        mean=("value", "mean"), sd=("value", "std")
    )
    mean_w = summary_long.pivot(index="condition_id", columns="metric", values="mean").add_suffix(
        "_mean"
    )
    sd_w = summary_long.pivot(index="condition_id", columns="metric", values="sd").add_suffix("_sd")
    summary = cond.merge(mean_w.reset_index(), on="condition_id").merge(
        sd_w.reset_index(), on="condition_id"
    )

    timing = experiment_result.timing_observations
    timing_fold = timing.groupby(["condition_id", "component", "fold"], as_index=False).seconds.mean()
    timing_long = timing_fold.groupby(["condition_id", "component"], as_index=False).agg(
        mean_seconds=("seconds", "mean"), sd_seconds=("seconds", "std")
    )
    timing_w = (
        timing_long.pivot(index="condition_id", columns="component", values="mean_seconds")
        .add_suffix("_mean")
        .reset_index()
    )
    summary = summary.merge(timing_w, on="condition_id", how="left", validate="one_to_one")
    return summary_long, by_fold, summary, timing_long, timing_fold


def build_model_overview(
    condition_summary: pd.DataFrame,
    primary_metrics: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Compute model-level median score, mean rank, and win count across main-matrix representations."""
    base = condition_summary.copy()
    if "execution_scope" in base:
        base = base[base.execution_scope == "main_matrix"].copy()
    parts = []
    for metric in primary_metrics:
        col = f"{metric}_mean"
        if col not in base:
            continue
        x = base[["representation_id", "model_id", "model_name", col]].copy()
        x["rank"] = x.groupby("representation_id")[col].rank(ascending=False, method="min")
        x["win"] = (
            x[col] == x.groupby("representation_id")[col].transform("max")
        ).astype(int)
        p = x.groupby(["model_id", "model_name"], as_index=False).agg(
            median=(col, "median"),
            mean_rank=("rank", "mean"),
            wins=("win", "sum"),
        )
        p["metric"] = metric
        parts.append(p)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def select_reference(
    condition_summary: pd.DataFrame,
    experiment_result: ExperimentResult,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rank conditions by harmonic mean of primary metrics minus stability penalty."""
    policy = experiment_result.metadata["reference_policy"]
    metrics = list(policy["metrics"])
    x = condition_summary.copy()
    vals = np.column_stack([x[f"{m}_mean"].clip(lower=0).to_numpy(float) for m in metrics])
    denom = np.sum(np.divide(1.0, vals, out=np.full_like(vals, np.inf), where=vals > 0), axis=1)
    hmean = np.divide(
        len(metrics),
        denom,
        out=np.zeros(len(x), dtype=float),
        where=np.isfinite(denom) & (denom > 0),
    )
    sd_sum = np.zeros(len(x), dtype=float)
    for m in metrics:
        if f"{m}_sd" in x:
            sd_sum += x[f"{m}_sd"].fillna(0).to_numpy(float)
    x["reference_score"] = hmean - float(policy.get("stability_weight", 0)) * sd_sum
    time_col = f"{policy.get('time_tiebreak_component', 'pipeline_incremental')}_mean"
    if time_col not in x:
        x[time_col] = np.nan
    x["_time_sort"] = x[time_col].fillna(np.inf)
    sort_cols = ["reference_score"] + [f"{m}_mean" for m in reversed(metrics)] + ["_time_sort"]
    asc = [False] * (1 + len(metrics)) + [True]
    ranking = x.sort_values(sort_cols, ascending=asc).drop(columns="_time_sort").reset_index(drop=True)
    return ranking, ranking.iloc[0].to_dict()


def build_reference_by_class(
    experiment_result: ExperimentResult,
    reference: dict[str, Any],
    dataset_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize per-class Precision, Recall, and F1 for the selected reference condition."""
    q = experiment_result.class_metric_observations
    q = q[q.condition_id == reference["condition_id"]]
    fold = q.groupby(["class_name", "metric", "fold"], as_index=False).value.mean()
    s = fold.groupby(["class_name", "metric"], as_index=False).agg(
        mean=("value", "mean"), sd=("value", "std")
    )
    mean_w = s.pivot(index="class_name", columns="metric", values="mean").add_suffix("_mean")
    sd_w = s.pivot(index="class_name", columns="metric", values="sd").add_suffix("_sd")
    out = mean_w.join(sd_w).reset_index()
    balance = dataset_tables["dataset_class_balance"][
        ["class_name", "support", "prevalence", "imbalance_ratio_vs_max"]
    ]
    return balance.merge(out, on="class_name", how="left", validate="one_to_one")


def pareto_front(df: pd.DataFrame, score_col: str, time_col: str) -> pd.DataFrame:
    """Identify non-dominated conditions maximizing score and minimizing execution time."""
    x = df.dropna(subset=[score_col, time_col]).reset_index(drop=True)
    keep = []
    vals = x[[score_col, time_col]].to_numpy(float)
    for i, (score_i, time_i) in enumerate(vals):
        dominated = any(
            j != i and sj >= score_i and tj <= time_i and (sj > score_i or tj < time_i)
            for j, (sj, tj) in enumerate(vals)
        )
        if not dominated:
            keep.append(i)
    return x.iloc[keep].sort_values(time_col).reset_index(drop=True)


def build_plot_specs(
    tables: dict[str, pd.DataFrame],
    reference: dict[str, Any],
    experiment_result: ExperimentResult,
    diagnostics: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Construct declarative specifications for all figures rendered by render_study."""
    labels = experiment_result.metadata["metric_labels"]
    primary = list(experiment_result.metadata["primary_metrics"])
    p0, p1 = primary[:2]
    specs: list[dict[str, Any]] = [
        {
            "id": "dataset_support",
            "kind": "barh",
            "table": "dataset_class_balance",
            "x": "support",
            "y": "class_name",
            "title": "Suporte por macrogênero — desenvolvimento",
            "xlabel": "Número de obras",
            "ylabel": "Macrogênero",
        },
        {
            "id": "dataset_cardinality",
            "kind": "bar",
            "table": "dataset_label_cardinality",
            "x": "labels_per_item",
            "y": "n_items",
            "title": "Cardinalidade multirrótulo",
            "xlabel": "Rótulos por obra",
            "ylabel": "Número de obras",
        },
        {
            "id": "dataset_jaccard",
            "kind": "heatmap",
            "table": "dataset_cooccurrence_jaccard",
            "title": "Coocorrência entre macrogêneros — Jaccard",
            "colorbar": "Jaccard",
            "vmin": 0,
            "vmax": 1,
        },
        {
            "id": "dataset_fold_support",
            "kind": "heatmap",
            "table": "dataset_fold_support_matrix",
            "title": "Suporte por macrogênero e fold",
            "colorbar": "Support",
            "annotate": True,
            "fmt": ".0f",
        },
        {
            "id": "top_conditions",
            "kind": "grouped_barh",
            "table": "reference_ranking_top",
            "y": "condition_label",
            "series": [f"{p0}_mean", f"{p1}_mean"],
            "series_labels": [labels.get(p0, p0), labels.get(p1, p1)],
            "title": "Condições mais fortes pela heurística de referência",
            "xlabel": "F1",
            "xlim": [0, 1],
        },
        {
            "id": "primary_scatter",
            "kind": "scatter",
            "table": "condition_summary",
            "x": f"{p0}_mean",
            "y": f"{p1}_mean",
            "group": "model_name",
            "label": "representation_name",
            "highlight": "is_reference",
            "label_filters": ["is_reference"],
            "title": f"{labels.get(p0, p0)} × {labels.get(p1, p1)} por condição",
            "xlabel": labels.get(p0, p0),
            "ylabel": labels.get(p1, p1),
            "auto_xlim": True,
            "auto_ylim": True,
            "x_bounds": [0, 1],
            "y_bounds": [0, 1],
        },
        {
            "id": "model_overview",
            "kind": "grouped_barh",
            "table": "model_overview_wide",
            "y": "model_name",
            "series": [f"{p0}_median", f"{p1}_median"],
            "series_labels": [labels.get(p0, p0), labels.get(p1, p1)],
            "title": "Visão agregada por modelo",
            "xlabel": "F1 mediano",
            "xlim": [0, 1],
        },
        {
            "id": "reference_by_class",
            "kind": "grouped_barh",
            "table": "reference_by_class",
            "y": "class_name",
            "series": ["precision_mean", "recall_mean", "f1_mean"],
            "series_labels": ["Precisão", "Recall", "F1"],
            "errors": [None, None, "f1_sd"],
            "title": f"Desempenho por macrogênero — referência: {reference['representation_name']} × {reference['model_name']}",
            "xlabel": "Métrica média (barra de erro no F1 = DP entre folds)",
            "xlim": [0, 1],
        },
        {
            "id": "support_vs_reference_f1",
            "kind": "scatter",
            "table": "reference_by_class",
            "x": "support",
            "y": "f1_mean",
            "label": "class_name",
            "title": "Suporte × F1 por macrogênero",
            "xlabel": "Support no desenvolvimento",
            "ylabel": "F1 médio",
            "trendline": True,
            "correlation": True,
            "auto_ylim": True,
            "y_bounds": [0, 1],
        },
        {
            "id": "score_vs_time",
            "kind": "scatter",
            "table": "score_time",
            "x": "pipeline_incremental_mean",
            "y": f"{p0}_mean",
            "group": "model_name",
            "label": "representation_name",
            "highlight": "is_reference",
            "label_filters": ["is_reference"],
            "line_filter": "is_pareto_primary",
            "xscale": "log",
            "title": "Desempenho × custo computacional",
            "xlabel": "Tempo incremental médio por fold (s, escala log)",
            "ylabel": labels.get(p0, p0),
            "auto_ylim": True,
            "y_bounds": [0, 1],
        },
    ]

    if "graph_operator_comparison" in diagnostics and not diagnostics["graph_operator_comparison"].empty:
        specs.append({
            "id": "graph_operator_comparison",
            "kind": "grouped_barh",
            "table": "diagnostic__graph_operator_comparison",
            "y": "condition_label",
            "series": [f"{p0}_mean", f"{p1}_mean"],
            "series_labels": [labels.get(p0, p0), labels.get(p1, p1)],
            "errors": [f"{p0}_sd", f"{p1}_sd"],
            "title": "Operadores homogêneos — MiniLM + word_1 no mesmo kNN",
            "xlabel": "F1 médio (barra de erro = DP entre folds)",
            "xlim": [0, 1],
        })

    if "hetero_structural_comparison" in diagnostics and not diagnostics["hetero_structural_comparison"].empty:
        specs.append({
            "id": "hetero_structural_comparison",
            "kind": "grouped_barh",
            "table": "diagnostic__hetero_structural_comparison",
            "y": "condition_label",
            "series": [f"{p0}_mean", f"{p1}_mean"],
            "series_labels": [labels.get(p0, p0), labels.get(p1, p1)],
            "errors": [f"{p0}_sd", f"{p1}_sd"],
            "title": "Heterogeneidade — densidade lexical e atenção termo→obra",
            "xlabel": "F1 médio (barra de erro = DP entre folds)",
            "xlim": [0, 1],
        })

    if "learning_curve__micro" in diagnostics:
        specs.append({
            "id": "learning_curve_micro",
            "kind": "line_error",
            "table": "diagnostic__learning_curve__micro",
            "x": "fraction_percent",
            "y": "mean",
            "error": "sd",
            "group": "model_name",
            "title": "Learning curve — Micro-F1",
            "xlabel": "Percentual do treino externo",
            "ylabel": labels.get(p0, p0),
            "auto_ylim": True,
            "y_bounds": [0, 1],
            "xtick_percent": True,
        })
    if "learning_curve__macro" in diagnostics:
        specs.append({
            "id": "learning_curve_macro",
            "kind": "line_error",
            "table": "diagnostic__learning_curve__macro",
            "x": "fraction_percent",
            "y": "mean",
            "error": "sd",
            "group": "model_name",
            "title": "Learning curve — Macro-F1",
            "xlabel": "Percentual do treino externo",
            "ylabel": labels.get(p1, p1),
            "auto_ylim": True,
            "y_bounds": [0, 1],
            "xtick_percent": True,
        })
    if "learning_curve__class_heatmap" in diagnostics:
        gain_view = diagnostics.get("learning_curve__class_gain", pd.DataFrame())
        focus_name = (
            str(gain_view.model_name.iloc[0])
            if not gain_view.empty and "model_name" in gain_view
            else "modelo relacional"
        )
        specs.append({
            "id": "learning_curve_by_class",
            "kind": "heatmap",
            "table": "diagnostic__learning_curve__class_heatmap",
            "title": f"Learning curve por macrogênero — {focus_name}",
            "colorbar": "F1",
            "vmin": 0,
            "vmax": 1,
            "annotate": True,
            "fmt": ".2f",
            "xlabel": "Percentual do treino externo",
            "ylabel": "Macrogênero",
        })
    if "learning_curve__class_gain" in diagnostics:
        specs.append({
            "id": "learning_curve_class_gain",
            "kind": "barh",
            "table": "diagnostic__learning_curve__class_gain",
            "x": "delta_f1",
            "y": "class_name",
            "title": "Ganho de F1 por macrogênero — 20% → 100% do treino",
            "xlabel": "Δ F1",
            "ylabel": "Macrogênero",
            "vline": 0,
        })

    if (
        "graph_samples__samples" in diagnostics
        and "graph_samples__nodes" in diagnostics
        and "graph_samples__edges" in diagnostics
    ):
        sample_meta = diagnostics["graph_samples__samples"]
        for row in sample_meta.itertuples(index=False):
            sample_id = str(row.sample_id)
            if str(row.color_mode) == "focus_class":
                specs.append({
                    "id": sample_id,
                    "kind": "network",
                    "table": "diagnostic__graph_samples__nodes",
                    "edges_table": "diagnostic__graph_samples__edges",
                    "filters": {"sample_id": sample_id},
                    "group": "focus_group",
                    "label": "display_name",
                    "label_flag": "show_label",
                    "center": "is_center",
                    "edge_weight": "cosine",
                    "title": str(row.title),
                    "subtitle": f"Fold {int(row.fold)} | {int(row.n_nodes)} nós | {int(row.n_edges)} arestas",
                })
            else:
                specs.append({
                    "id": sample_id,
                    "kind": "network",
                    "table": "diagnostic__graph_samples__nodes",
                    "edges_table": "diagnostic__graph_samples__edges",
                    "filters": {"sample_id": sample_id},
                    "color_numeric": "label_jaccard_to_center",
                    "colorbar": "Jaccard de rótulos com o centro",
                    "label": "display_name",
                    "label_flag": "show_label",
                    "center": "is_center",
                    "edge_weight": "cosine",
                    "title": str(row.title),
                    "subtitle": f"Fold {int(row.fold)} | {int(row.n_nodes)} nós | {int(row.n_edges)} arestas",
                })

    for diag_id, df in diagnostics.items():
        if (
            diag_id.endswith("__edges")
            or diag_id.startswith("graph_samples__")
            or df.empty
            or not {"mean_jaccard", "delta", "representation_name"}.issubset(df.columns)
        ):
            continue
        specs.append({
            "id": diag_id,
            "kind": "scatter",
            "table": f"diagnostic__{diag_id}",
            "x": "mean_jaccard",
            "y": "delta",
            "label": "representation_name",
            "title": str(df.comparison_label.iloc[0]) + " — homofilia × ganho",
            "xlabel": "Jaccard médio entre vizinhos",
            "ylabel": "Δ da métrica relacional",
            "hline": 0,
            "trendline": True,
            "correlation": True,
            "auto_xlim": True,
            "auto_ylim": True,
        })
    return specs


def assemble_study_result(
    dataset: DatasetContext,
    experiment_result: ExperimentResult,
    diagnostics: dict[str, pd.DataFrame] | None = None,
) -> StudyResult:
    """Assemble final StudyResult contract combining dataset profile, experiment summaries, and diagnostics."""
    diagnostics = diagnostics or {}
    tables = build_dataset_profile(dataset)
    summary_long, by_fold, condition_summary, timing_long, timing_by_fold = summarize_experiment(
        experiment_result
    )
    ranking, reference = select_reference(condition_summary, experiment_result)
    reference_by_class = build_reference_by_class(experiment_result, reference, tables)
    model_overview = build_model_overview(
        condition_summary, experiment_result.metadata["primary_metrics"]
    )
    model_overview_wide = (
        model_overview.pivot(index=["model_id", "model_name"], columns="metric", values="median")
        .add_suffix("_median")
        .reset_index()
        if not model_overview.empty
        else pd.DataFrame()
    )

    primary = experiment_result.metadata["primary_metrics"][0]
    score_time = condition_summary.copy()
    score_time["is_reference"] = score_time.condition_id == reference["condition_id"]
    if "pipeline_incremental_mean" in score_time:
        pareto_primary = pareto_front(score_time, f"{primary}_mean", "pipeline_incremental_mean")
        pareto_ids = set(pareto_primary.condition_id)
        score_time["is_pareto_primary"] = score_time.condition_id.isin(pareto_ids)
    else:
        pareto_primary = pd.DataFrame()
        score_time["is_pareto_primary"] = False

    condition_summary["is_reference"] = condition_summary.condition_id == reference["condition_id"]
    ranking["condition_label"] = (
        ranking.representation_name.astype(str) + " × " + ranking.model_name.astype(str)
    )
    ranking_top = ranking.head(15).sort_values("reference_score").reset_index(drop=True)

    tables.update({
        "conditions": experiment_result.conditions,
        "metric_summary_long": summary_long,
        "metric_by_fold": by_fold,
        "condition_summary": condition_summary,
        "timing_summary_long": timing_long,
        "timing_by_fold": timing_by_fold,
        "model_overview": model_overview,
        "model_overview_wide": model_overview_wide,
        "reference_ranking": ranking,
        "reference_ranking_top": ranking_top,
        "reference_by_class": reference_by_class,
        "score_time": score_time,
        "pareto_primary": pareto_primary,
    })
    for k, v in diagnostics.items():
        tables[f"diagnostic__{k}"] = v

    table_specs: list[dict[str, Any]] = [
        {"table": "dataset_overview", "title": "Dados de desenvolvimento"},
        {"table": "dataset_class_balance", "title": "Balanceamento por macrogênero"},
        {"table": "reference_ranking", "title": "Top condições pela heurística", "limit": 15},
        {"table": "reference_by_class", "title": "Métricas por macrogênero da condição de referência"},
        {"table": "model_overview", "title": "Visão agregada dos modelos"},
        {"table": "pareto_primary", "title": "Fronteira de Pareto da métrica primária"},
    ]
    if "graph_operator_comparison" in diagnostics and not diagnostics["graph_operator_comparison"].empty:
        table_specs.append({
            "table": "diagnostic__graph_operator_comparison",
            "title": "Operadores homogêneos — comparação focal",
        })
    if "hetero_structural_comparison" in diagnostics and not diagnostics["hetero_structural_comparison"].empty:
        table_specs.append({
            "table": "diagnostic__hetero_structural_comparison",
            "title": "Experimento heterogêneo — comparação focal",
        })
    if "hetero_term_structure" in diagnostics and not diagnostics["hetero_term_structure"].empty:
        table_specs.append({
            "table": "diagnostic__hetero_term_structure",
            "title": "Grafo heterogêneo — estrutura por fold",
        })
    if "learning_curve" in diagnostics:
        table_specs.append({
            "table": "diagnostic__learning_curve",
            "title": "Learning curve — resumo global",
        })
    if "learning_curve__class_gain" in diagnostics:
        table_specs.append({
            "table": "diagnostic__learning_curve__class_gain",
            "title": "Learning curve — ganho por macrogênero",
        })
    if "graph_samples__samples" in diagnostics:
        table_specs.append({
            "table": "diagnostic__graph_samples__samples",
            "title": "Amostras qualitativas dos grafos",
        })

    plots = build_plot_specs(tables, reference, experiment_result, diagnostics)
    metadata = {
        "study_version": STUDY_VERSION,
        "experiment_schema": experiment_result.schema_version,
        "primary_metrics": experiment_result.metadata["primary_metrics"],
        "metric_labels": experiment_result.metadata["metric_labels"],
        "reference_policy": experiment_result.metadata["reference_policy"],
        "dataset": dataset.metadata,
        "audit_used": False,
        "analysis_version": "agnostic_renderer_v4",
        "diagnostics": sorted(diagnostics.keys()),
        "learning_curve": LEARNING_CURVE_CONFIG if "learning_curve" in diagnostics else None,
        "graph_visualization": GRAPH_VIS_CONFIG if "graph_samples__samples" in diagnostics else None,
    }
    return StudyResult(RESULT_SCHEMA_VERSION, metadata, tables, reference, table_specs, plots)
