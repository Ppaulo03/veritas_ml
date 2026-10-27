"""Framework-agnostic experimental execution engine, incremental reuse, and result persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import (
    CENTRAL,
    CV_SEED,
    EXPERIMENT_SCHEMA_VERSION,
    FIXED_THRESHOLD,
    INCREMENTAL_BASE_RUN_ID,
    INCREMENTAL_REUSE,
    INCREMENTAL_REUSE_STRICT,
    STUDY_DIR,
    STUDY_VERSION,
)
from ..core.contracts import (
    DatasetContext,
    ExperimentPlan,
    ExperimentResult,
    FeatureContext,
    GraphContext,
    stable_id,
)
from ..core.graph import (
    build_knn_edge_arrays,
    build_knn_propagation,
    prepare_hetero_term_incidence,
)
from ..features import WORD_CFGS, _fit_tfidf_base, bundle_component_time, combine_blocks
from ..models import class_metrics, multilabel_metrics, run_model


def _expected_rows_for_model(dataset: DatasetContext, plan: ExperimentPlan, model: str) -> int:
    return len(dataset.folds) * len(plan.model_seeds[model])


def _completed_conditions(
    df: pd.DataFrame | None,
    dataset: DatasetContext,
    plan: ExperimentPlan,
) -> set[tuple[str, str]]:
    if df is None or df.empty:
        return set()
    done: set[tuple[str, str]] = set()
    for (features, model), group in df.groupby(["features", "modelo"]):
        if len(group) == _expected_rows_for_model(dataset, plan, str(model)):
            done.add((str(features), str(model)))
    return done


def _condition_table_from_raw(raw_metrics: pd.DataFrame, plan: ExperimentPlan) -> pd.DataFrame:
    cols = ["family", "features", "fontes", "n_fontes", "input_dim", "modelo"]
    x = raw_metrics[cols].drop_duplicates().copy()
    x["representation_id"] = x.features.map(lambda v: stable_id("repr", v))
    x["condition_id"] = [
        stable_id("cond", f"{r.features}|{r.modelo}") for r in x.itertuples(index=False)
    ]
    x["representation_name"] = x.features.astype(str)
    x["model_id"] = x.modelo.astype(str)
    x["model_name"] = x.model_id.map(
        lambda m: plan.model_metadata.get(m, {}).get("display_name", m)
    )
    x["model_family"] = x.model_id.map(
        lambda m: plan.model_metadata.get(m, {}).get("family", "unknown")
    )
    x["model_role"] = x.model_id.map(
        lambda m: plan.model_metadata.get(m, {}).get("role", "unknown")
    )
    x["execution_scope"] = x.model_id.map(
        lambda m: plan.model_metadata.get(m, {}).get("execution_scope", "main_matrix")
    )
    x["model_enabled"] = x.model_id.map(
        lambda m: bool(plan.model_metadata.get(m, {}).get("enabled", True))
    )
    out_cols = [
        "condition_id",
        "representation_id",
        "representation_name",
        "model_id",
        "model_name",
        "model_family",
        "model_role",
        "execution_scope",
        "model_enabled",
        "family",
        "fontes",
        "n_fontes",
        "input_dim",
    ]
    return x[out_cols].sort_values(["representation_name", "model_id"]).reset_index(drop=True)


def standardize_experiment_result(
    raw_metrics: pd.DataFrame,
    raw_class_metrics: pd.DataFrame,
    plan: ExperimentPlan,
) -> ExperimentResult:
    """Convert raw execution frames into standardized relational observation tables."""
    conditions = _condition_table_from_raw(raw_metrics, plan)
    key_map = conditions.set_index(["representation_name", "model_id"]).condition_id.to_dict()

    m = raw_metrics.copy()
    m["condition_id"] = [key_map[(str(f), str(mod))] for f, mod in zip(m.features, m.modelo)]
    metric_cols = [c for c in plan.reported_metrics if c in m.columns]
    metric_obs = m.melt(
        id_vars=["condition_id", "fold", "seed"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )

    cm = raw_class_metrics.copy()
    cm["condition_id"] = [key_map[(str(f), str(mod))] for f, mod in zip(cm.features, cm.modelo)]
    class_cols = [c for c in plan.class_metrics if c in cm.columns]
    class_obs = cm.melt(
        id_vars=["condition_id", "fold", "seed", "class_name"],
        value_vars=class_cols,
        var_name="metric",
        value_name="value",
    )

    time_map = {
        "feature_fit_seconds_incremental": "feature_fit",
        "feature_transform_seconds_incremental": "feature_transform",
        "feature_fusion_seconds": "feature_fusion",
        "graph_build_seconds": "graph_build",
        "train_seconds": "train",
        "predict_seconds": "predict",
        "model_seconds": "model",
        "pipeline_seconds_incremental": "pipeline_incremental",
        "condition_wall_seconds_observed": "condition_wall",
    }
    present = {k: v for k, v in time_map.items() if k in m.columns}
    timing = m.melt(
        id_vars=["condition_id", "fold", "seed"],
        value_vars=list(present),
        var_name="_raw_component",
        value_name="seconds",
    )
    timing["component"] = timing._raw_component.map(present)
    timing = timing.drop(columns="_raw_component")

    return ExperimentResult(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        conditions=conditions,
        metric_observations=metric_obs,
        class_metric_observations=class_obs,
        timing_observations=timing,
        raw_metrics=raw_metrics,
        raw_class_metrics=raw_class_metrics,
        metadata={
            "primary_metrics": list(plan.primary_metrics),
            "metric_labels": dict(plan.metric_labels),
            "reference_policy": dict(plan.reference_policy),
            "model_metadata": plan.model_metadata,
        },
    )


def persist_experiment_result(result: ExperimentResult, directory: str | Path) -> None:
    """Persist standardized experiment observation CSVs to disk."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.conditions.to_csv(out_dir / "conditions.csv", index=False)
    result.metric_observations.to_csv(out_dir / "metric_observations.csv", index=False)
    result.class_metric_observations.to_csv(out_dir / "class_metric_observations.csv", index=False)
    result.timing_observations.to_csv(out_dir / "timing_observations.csv", index=False)


def _current_training_signature(plan: ExperimentPlan) -> dict[str, Any]:
    return {
        "threshold": FIXED_THRESHOLD,
        "models": {
            m: meta.get("training_config", {})
            for m, meta in plan.model_metadata.items()
        },
    }


def _allowed_condition_pairs(plan: ExperimentPlan) -> set[tuple[str, str]]:
    allowed = {(str(e["features"]), str(m)) for e in plan.registry for m in plan.model_ids}
    allowed.update(
        (str(e["features"]), str(e["model_id"])) for e in plan.special_experiments
    )
    return allowed


def load_incremental_base(
    dataset: DatasetContext,
    plan: ExperimentPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load compatible base run checkpoints when incremental reuse is enabled."""
    if not INCREMENTAL_REUSE or not INCREMENTAL_BASE_RUN_ID:
        print("Reuso incremental desabilitado; execução integral solicitada.")
        return pd.DataFrame(), pd.DataFrame()

    base = CENTRAL / "runs" / INCREMENTAL_BASE_RUN_ID / "estudo_unificado"
    manifest_path = base / "manifesto.json"
    metrics_path = base / "_internal_metrics_partial.csv"
    class_path = base / "_internal_class_metrics_partial.csv"
    print(f"Base incremental solicitada: {base}")

    missing = [str(p) for p in (manifest_path, metrics_path, class_path) if not p.is_file()]
    if missing:
        msg = "Checkpoint incremental incompleto/não encontrado: " + "; ".join(missing)
        if INCREMENTAL_REUSE_STRICT:
            raise FileNotFoundError(msg)
        print(msg + "; execução seguirá sem reuso.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ds = manifest.get("dataset", {})
        checks = {
            "dataset_hash": ds.get("data_sha256") == dataset.metadata.get("data_sha256"),
            "split_hash": ds.get("split_sha256") == dataset.metadata.get("split_sha256"),
            "audit_unused": manifest.get("audit_used") is False,
            "cv_splits": manifest.get("cv", {}).get("outer_splits") == len(dataset.folds),
            "fold_seed": manifest.get("cv", {}).get("fold_seed") == CV_SEED,
            "training": manifest.get("training") == _current_training_signature(plan),
            "graph": manifest.get("graph") == plan.graph_config,
        }
        print(
            "Compatibilidade incremental:",
            ", ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items()),
        )
        if not all(checks.values()):
            bad = [k for k, v in checks.items() if not v]
            msg = "Checkpoint incremental rejeitado por incompatibilidade: " + ", ".join(bad)
            if INCREMENTAL_REUSE_STRICT:
                raise RuntimeError(msg)
            print(msg + "; execução seguirá sem reuso.")
            return pd.DataFrame(), pd.DataFrame()

        raw = pd.read_csv(metrics_path)
        cls = pd.read_csv(class_path)
        base_pairs = raw[["features", "modelo"]].drop_duplicates().shape[0]
        base_counts = raw.groupby(["features", "modelo"]).size()
        malformed = base_counts[base_counts != len(dataset.folds) * 3]
        if len(malformed):
            print(
                f"Aviso: {len(malformed)} pares da base não têm 15 linhas; serão ignorados se incompletos."
            )

        allowed = _allowed_condition_pairs(plan)
        mask = [(str(f), str(m)) in allowed for f, m in zip(raw.features, raw.modelo)]
        raw = raw.loc[mask].copy()
        done = _completed_conditions(raw, dataset, plan)
        raw = raw[[(str(f), str(m)) in done for f, m in zip(raw.features, raw.modelo)]].copy()
        if not cls.empty:
            cls = cls[[(str(f), str(m)) in done for f, m in zip(cls.features, cls.modelo)]].copy()

        if not done:
            msg = f"Nenhuma condição completa reutilizável encontrada na base ({base_pairs} pares brutos)."
            if INCREMENTAL_REUSE_STRICT:
                raise RuntimeError(msg)
            print(msg + "; execução seguirá sem reuso.")
            return pd.DataFrame(), pd.DataFrame()

        raw["study_version"] = STUDY_VERSION
        if not cls.empty and "study_version" in cls.columns:
            cls["study_version"] = STUDY_VERSION
        print(
            f"Reuso incremental ATIVO: {len(done)} condições completas reaproveitadas de {INCREMENTAL_BASE_RUN_ID}."
        )
        return raw, cls
    except Exception:
        if INCREMENTAL_REUSE_STRICT:
            raise
        print("Checkpoint incremental rejeitado; execução seguirá sem reuso.")
        return pd.DataFrame(), pd.DataFrame()


def execute_experiments(
    dataset: DatasetContext,
    feature_context: FeatureContext,
    plan: ExperimentPlan,
) -> ExperimentResult:
    """Execute the full experimental matrix and focal conditions via the pluggable model registry."""
    metrics_path = STUDY_DIR / "_internal_metrics_partial.csv"
    class_path = STUDY_DIR / "_internal_class_metrics_partial.csv"

    if metrics_path.is_file():
        raw_metrics = pd.read_csv(metrics_path)
        if "study_version" not in raw_metrics or set(raw_metrics.study_version) != {STUDY_VERSION}:
            raw_metrics = pd.DataFrame()
        raw_class = pd.read_csv(class_path) if class_path.is_file() else pd.DataFrame()
    else:
        raw_metrics, raw_class = load_incremental_base(dataset, plan)
        if not raw_metrics.empty:
            raw_metrics.to_csv(metrics_path, index=False)
            raw_class.to_csv(class_path, index=False)

    completed = _completed_conditions(raw_metrics, dataset, plan)
    total = len(plan.registry) * len(plan.model_ids) + len(plan.special_experiments)
    current = 0
    print(
        f"Plano incremental: {len(completed)} condição(ões) já completas; "
        f"{total - len(completed)} condição(ões) pendentes nesta versão."
    )

    for entry in plan.registry:
        bundle = entry["bundle"]
        representation = entry["features"]
        family = entry["family"]
        pending = [m for m in plan.model_ids if (representation, m) not in completed]
        if not pending:
            for model in plan.model_ids:
                current += 1
                print(f"[{current:03d}/{total}] {representation} × {model}: checkpoint")
            continue

        fold_arrays = []
        dims = []
        for (tr, va), pack in zip(dataset.folds, feature_context.fold_packs):
            t0 = time.perf_counter()
            xtr = combine_blocks(pack["train"], bundle)
            xva = combine_blocks(pack["valid"], bundle)
            fusion_s = time.perf_counter() - t0
            feature_fit_s, feature_transform_s, components = bundle_component_time(pack, bundle)
            fold_arrays.append(
                (tr, va, xtr, xva, feature_fit_s, feature_transform_s, fusion_s, components)
            )
            dims.append(xtr.shape[1])

        if len(set(dims)) != 1:
            raise AssertionError(f"Dimensão instável em {representation}: {dims}")
        input_dim = dims[0]

        for model in plan.model_ids:
            current += 1
            if (representation, model) in completed:
                print(f"[{current:03d}/{total}] {representation} × {model}: checkpoint")
                continue

            rows = []
            crows = []
            condition_start = time.perf_counter()

            for fold, (tr, va, xtr, xva, feature_fit_s, feature_transform_s, fusion_s, components) in enumerate(
                fold_arrays, start=1
            ):
                graph_s = 0.0
                graph_ctx = None
                if plan.model_metadata.get(model, {}).get("role") == "relational":
                    tg = time.perf_counter()
                    p_train, p_eval, edges_df = build_knn_propagation(xtr, xva, **plan.graph_config)
                    graph_s = time.perf_counter() - tg
                    graph_ctx = GraphContext(p_train=p_train, p_eval=p_eval, edges_df=edges_df)

                for seed in plan.model_seeds[model]:
                    prob, complexity, kind, train_s, predict_s = run_model(
                        model, xtr, dataset.targets[tr], xva, graph_ctx, seed
                    )
                    score = multilabel_metrics(dataset.targets[va], prob)
                    pipeline_s = (
                        feature_fit_s + feature_transform_s + fusion_s + graph_s + train_s + predict_s
                    )
                    rows.append({
                        "study_version": STUDY_VERSION,
                        "family": family,
                        "features": representation,
                        "fontes": "|".join(bundle),
                        "n_fontes": len(bundle),
                        "input_dim": input_dim,
                        "modelo": model,
                        "fold": fold,
                        "seed": seed,
                        "n_treino": len(tr),
                        "n_validacao": len(va),
                        "model_complexity": complexity,
                        "complexity_kind": kind,
                        "feature_fit_seconds_incremental": feature_fit_s,
                        "feature_transform_seconds_incremental": feature_transform_s,
                        "feature_fusion_seconds": fusion_s,
                        "feature_components": "|".join(components),
                        "graph_build_seconds": graph_s,
                        "train_seconds": train_s,
                        "predict_seconds": predict_s,
                        "model_seconds": train_s + predict_s,
                        "pipeline_seconds_incremental": pipeline_s,
                        **score,
                    })
                    cm = class_metrics(dataset.targets[va], prob, dataset.classes)
                    cm.insert(0, "seed", seed)
                    cm.insert(0, "fold", fold)
                    cm.insert(0, "modelo", model)
                    cm.insert(0, "features", representation)
                    cm.insert(0, "family", family)
                    crows.append(cm)

            wall = time.perf_counter() - condition_start
            for row in rows:
                row["condition_wall_seconds_observed"] = wall

            raw_metrics = pd.concat([raw_metrics, pd.DataFrame(rows)], ignore_index=True)
            raw_class = pd.concat([raw_class, *crows], ignore_index=True) if crows else raw_class
            raw_metrics.to_csv(metrics_path, index=False)
            raw_class.to_csv(class_path, index=False)
            m = pd.DataFrame(rows)
            print(
                f"[{current:03d}/{total}] {representation} × {model} | {wall:6.1f}s | "
                f"micro={m.f1_micro.mean():.3f} macro={m.f1_macro.mean():.3f}"
            )

    # Execute special structural / focal experiments
    for spec in plan.special_experiments:
        current += 1
        representation = str(spec["features"])
        model = str(spec["model_id"])
        family = str(spec["family"])
        adapter = str(spec.get("adapter", ""))

        if (representation, model) in completed:
            print(f"[{current:03d}/{total}] {representation} × {model}: checkpoint")
            continue

        rows = []
        crows = []
        condition_start = time.perf_counter()

        if adapter == "homogeneous_operator":
            bundle = tuple(spec["bundle"])
            for fold, ((tr, va), pack) in enumerate(
                zip(dataset.folds, feature_context.fold_packs), start=1
            ):
                t0 = time.perf_counter()
                xtr = combine_blocks(pack["train"], bundle)
                xva = combine_blocks(pack["valid"], bundle)
                fusion_s = time.perf_counter() - t0
                feature_fit_s, feature_transform_s, components = bundle_component_time(pack, bundle)

                tg = time.perf_counter()
                edge_train = build_knn_edge_arrays(
                    xtr,
                    np.empty((0, xtr.shape[1]), dtype=np.float32),
                    k=plan.graph_config["k"],
                    min_similarity=plan.graph_config.get("min_similarity"),
                )
                edge_eval = build_knn_edge_arrays(
                    xtr,
                    xva,
                    k=plan.graph_config["k"],
                    min_similarity=plan.graph_config.get("min_similarity"),
                )
                graph_s = time.perf_counter() - tg
                graph_ctx = GraphContext(edge_index_train=edge_train, edge_index_eval=edge_eval)

                for seed in plan.model_seeds[model]:
                    prob, complexity, kind, train_s, predict_s = run_model(
                        model, xtr, dataset.targets[tr], xva, graph_ctx, seed
                    )
                    score = multilabel_metrics(dataset.targets[va], prob)
                    pipeline_s = (
                        feature_fit_s + feature_transform_s + fusion_s + graph_s + train_s + predict_s
                    )
                    rows.append({
                        "study_version": STUDY_VERSION,
                        "family": family,
                        "features": representation,
                        "fontes": spec["fontes"],
                        "n_fontes": int(spec["n_fontes"]),
                        "input_dim": int(xtr.shape[1]),
                        "modelo": model,
                        "fold": fold,
                        "seed": seed,
                        "n_treino": len(tr),
                        "n_validacao": len(va),
                        "model_complexity": complexity,
                        "complexity_kind": kind,
                        "feature_fit_seconds_incremental": feature_fit_s,
                        "feature_transform_seconds_incremental": feature_transform_s,
                        "feature_fusion_seconds": fusion_s,
                        "feature_components": "|".join(components),
                        "graph_build_seconds": graph_s,
                        "train_seconds": train_s,
                        "predict_seconds": predict_s,
                        "model_seconds": train_s + predict_s,
                        "pipeline_seconds_incremental": pipeline_s,
                        "graph_k": int(plan.graph_config["k"]),
                        **score,
                    })
                    cm = class_metrics(dataset.targets[va], prob, dataset.classes)
                    cm.insert(0, "seed", seed)
                    cm.insert(0, "fold", fold)
                    cm.insert(0, "modelo", model)
                    cm.insert(0, "features", representation)
                    cm.insert(0, "family", family)
                    crows.append(cm)

        elif adapter == "hetero_term":
            top_k = spec.get("top_k_terms")
            for fold, ((tr, va), pack) in enumerate(
                zip(dataset.folds, feature_context.fold_packs), start=1
            ):
                texts_tr = dataset.frame.iloc[tr].texto.tolist()
                texts_va = dataset.frame.iloc[va].texto.tolist()
                _, a_tr, a_va, fit_s, transform_s = _fit_tfidf_base(
                    texts_tr, texts_va, WORD_CFGS["word_1"]
                )
                tg = time.perf_counter()
                row_tr, col_tr, row_va, graph_meta = prepare_hetero_term_incidence(
                    a_tr, a_va, top_k=top_k
                )
                graph_s = time.perf_counter() - tg
                graph_ctx = GraphContext(
                    row_norm_train=row_tr,
                    col_norm_train=col_tr,
                    row_norm_eval=row_va,
                    metadata=graph_meta,
                )
                xtr = pack["train"]["minilm"]
                xva = pack["valid"]["minilm"]
                input_dim = int(xtr.shape[1])

                for seed in plan.model_seeds[model]:
                    prob, complexity, kind, train_s, predict_s = run_model(
                        model, xtr, dataset.targets[tr], xva, graph_ctx, seed
                    )
                    score = multilabel_metrics(dataset.targets[va], prob)
                    pipeline_s = fit_s + transform_s + graph_s + train_s + predict_s
                    rows.append({
                        "study_version": STUDY_VERSION,
                        "family": family,
                        "features": representation,
                        "fontes": spec["fontes"],
                        "n_fontes": int(spec["n_fontes"]),
                        "input_dim": input_dim,
                        "modelo": model,
                        "fold": fold,
                        "seed": seed,
                        "n_treino": len(tr),
                        "n_validacao": len(va),
                        "model_complexity": complexity,
                        "complexity_kind": kind,
                        "feature_fit_seconds_incremental": fit_s,
                        "feature_transform_seconds_incremental": transform_s,
                        "feature_fusion_seconds": 0.0,
                        "feature_components": "word_1_term_incidence",
                        "graph_build_seconds": graph_s,
                        "train_seconds": train_s,
                        "predict_seconds": predict_s,
                        "model_seconds": train_s + predict_s,
                        "pipeline_seconds_incremental": pipeline_s,
                        "term_nodes": graph_meta["term_nodes"],
                        "incidence_train": graph_meta["incidence_train"],
                        "incidence_valid": graph_meta["incidence_eval"],
                        "zero_term_valid_rows": graph_meta["zero_term_eval_rows"],
                        "term_top_k": (
                            np.nan
                            if graph_meta["term_top_k"] is None
                            else int(graph_meta["term_top_k"])
                        ),
                        **score,
                    })
                    cm = class_metrics(dataset.targets[va], prob, dataset.classes)
                    cm.insert(0, "seed", seed)
                    cm.insert(0, "fold", fold)
                    cm.insert(0, "modelo", model)
                    cm.insert(0, "features", representation)
                    cm.insert(0, "family", family)
                    crows.append(cm)
        else:
            raise ValueError(f"Experimento especial sem adapter: {spec}")

        wall = time.perf_counter() - condition_start
        for row in rows:
            row["condition_wall_seconds_observed"] = wall

        raw_metrics = pd.concat([raw_metrics, pd.DataFrame(rows)], ignore_index=True)
        raw_class = pd.concat([raw_class, *crows], ignore_index=True) if crows else raw_class
        raw_metrics.to_csv(metrics_path, index=False)
        raw_class.to_csv(class_path, index=False)
        m = pd.DataFrame(rows)
        suffix = ""
        if "term_nodes" in m:
            suffix = (
                f" | termos≈{int(m.term_nodes.mean())} | incidências≈{int(m.incidence_train.mean())}"
            )
        print(
            f"[{current:03d}/{total}] {representation} × {model} | {wall:6.1f}s | "
            f"micro={m.f1_micro.mean():.3f} macro={m.f1_macro.mean():.3f}{suffix}"
        )

    result = standardize_experiment_result(raw_metrics, raw_class, plan)
    persist_experiment_result(result, STUDY_DIR)

    expected = sum(_expected_rows_for_model(dataset, plan, m) for m in plan.model_ids) * len(
        plan.registry
    )
    expected += sum(
        _expected_rows_for_model(dataset, plan, e["model_id"]) for e in plan.special_experiments
    )
    assert len(raw_metrics) == expected, (len(raw_metrics), expected)
    return result
