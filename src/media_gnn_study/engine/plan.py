"""Experimental bundle catalog, focal relational conditions, and ExperimentPlan assembly."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from ..core.config import (
    CLASS_METRICS,
    GRAPH_CFG,
    HETERO_DIFFUSION_CFG,
    HETERO_TERM_CFG,
    METRIC_LABELS,
    MODEL_ENABLED,
    MODEL_METADATA,
    MODEL_SEEDS,
    MAIN_MODEL_ORDER,
    PRIMARY_METRICS,
    REFERENCE_POLICY,
    RELATIONAL_DIAGNOSTICS,
    REPORTED_METRICS,
    STUDY_DIR,
    ensure_runtime_dirs,
)
from ..core.contracts import DatasetContext, ExperimentPlan
from ..features import SOURCE_ORDER


def _dedupe_registry(rows: list[tuple[str, tuple[str, ...]]]) -> list[dict[str, Any]]:
    seen: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for family, bundle in rows:
        bundle_tuple = tuple(bundle)
        key = "+".join(bundle_tuple)
        if key not in seen:
            seen[key] = family
            out.append({"family": family, "bundle": bundle_tuple, "features": key})
    return out


CORE_SOURCES = ("minilm", "word_1_2", "entities", "emotion", "topics")

_registry_rows: list[tuple[str, tuple[str, ...]]] = []
for r in range(1, len(CORE_SOURCES) + 1):
    for combo in combinations(CORE_SOURCES, r):
        _registry_rows.append(("core_factorial", combo))

_registry_rows += [
    ("lexical", ("word_1",)),
    ("lexical", ("word_1_3",)),
    ("lexical", ("char_3_5",)),
    ("lexical", ("word_1_2", "char_3_5")),
    ("lexical", ("word_1_3", "char_3_5")),
    ("lexical", ("minilm", "word_1")),
    ("lexical", ("minilm", "word_1_3")),
    ("lexical", ("minilm", "char_3_5")),
    ("lexical", ("minilm", "word_1_2", "char_3_5")),
    ("lexical", ("minilm", "word_1_3", "char_3_5")),
]

_registry_rows += [
    ("narrative", ("actions",)),
    ("narrative", ("linguistic",)),
    ("narrative", ("actions", "emotion")),
    ("narrative", ("actions", "topics")),
    ("narrative", ("actions", "emotion", "topics")),
    ("narrative", ("actions", "linguistic")),
    ("narrative", ("linguistic", "emotion")),
    ("narrative", ("linguistic", "topics")),
    ("narrative", ("minilm", "actions")),
    ("narrative", ("minilm", "linguistic")),
    ("narrative", ("minilm", "actions", "emotion")),
    ("narrative", ("minilm", "actions", "topics")),
    ("narrative", ("minilm", "actions", "linguistic")),
    ("narrative", ("minilm", "actions", "emotion", "topics")),
]

BUNDLE_REGISTRY = _dedupe_registry(_registry_rows)
for _row in BUNDLE_REGISTRY:
    assert all(s in SOURCE_ORDER for s in _row["bundle"])

BUNDLE_TABLE = pd.DataFrame([{**r, "bundle": "|".join(r["bundle"])} for r in BUNDLE_REGISTRY])

SPECIAL_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    # 1) Homogeneous operators: same features, same inductive kNN topology, different operators
    {
        "id": "homogeneous_gcn_minilm_word1_v1",
        "adapter": "homogeneous_operator",
        "family": "homogeneous_operator",
        "features": "minilm+word_1",
        "bundle": ("minilm", "word_1"),
        "model_id": "gcn",
        "fontes": "minilm|word_1",
        "n_fontes": 2,
        "enabled": True,
        "description": "GCNConv no mesmo kNN indutivo usado pelo baseline Fixed-kNN GNN",
    },
    {
        "id": "homogeneous_graphsage_minilm_word1_v1",
        "adapter": "homogeneous_operator",
        "family": "homogeneous_operator",
        "features": "minilm+word_1",
        "bundle": ("minilm", "word_1"),
        "model_id": "graphsage",
        "fontes": "minilm|word_1",
        "n_fontes": 2,
        "enabled": True,
        "description": "SAGEConv(mean) no mesmo kNN indutivo usado pelo baseline Fixed-kNN GNN",
    },
    {
        "id": "homogeneous_gatv2_minilm_word1_v1",
        "adapter": "homogeneous_operator",
        "family": "homogeneous_operator",
        "features": "minilm+word_1",
        "bundle": ("minilm", "word_1"),
        "model_id": "gatv2",
        "fontes": "minilm|word_1",
        "n_fontes": 2,
        "enabled": True,
        "description": "GATv2Conv no mesmo kNN indutivo; atenção aprende quanto confiar em cada vizinho",
    },
    # 2) Heterogeneous bipartite work↔term models & top-k pruning ablations
    {
        "id": "hetero_term_gnn_v1",
        "adapter": "hetero_term",
        "family": "heterogeneous",
        "features": HETERO_TERM_CFG["representation_name"],
        "model_id": "hetero_term_gnn",
        "fontes": "minilm|word_1_terms",
        "n_fontes": 2,
        "top_k_terms": None,
        "enabled": True,
        "description": "Baseline atual: MiniLM nas obras + todos os nós termo, incidência TF-IDF train-only",
    },
    *tuple(
        {
            "id": f"hetero_term_gnn_top{k}_v1",
            "adapter": "hetero_term",
            "family": "heterogeneous_diffusion",
            "features": f"{HETERO_TERM_CFG['representation_name']}_top{k}",
            "model_id": "hetero_term_gnn",
            "fontes": "minilm|word_1_terms",
            "n_fontes": 2,
            "top_k_terms": k,
            "enabled": True,
            "description": f"Mesma hetero, mantendo apenas os top-{k} termos TF-IDF por obra",
        }
        for k in HETERO_DIFFUSION_CFG["top_k_values"]
    ),
    {
        "id": "hetero_term_gatv2_v1",
        "adapter": "hetero_term",
        "family": "heterogeneous_attention",
        "features": HETERO_TERM_CFG["representation_name"],
        "model_id": "hetero_term_gatv2",
        "fontes": "minilm|word_1_terms",
        "n_fontes": 2,
        "top_k_terms": None,
        "enabled": True,
        "description": "Estados dos termos train-only; GATv2 aprende a agregação termo→obra usando TF-IDF como atributo de aresta",
    },
)


def build_experiment_plan(
    dataset: DatasetContext | None = None,
    persist_registry: bool = True,
) -> ExperimentPlan:
    """Assemble ExperimentPlan reflecting currently enabled main and focal models."""
    active_main_models = tuple(
        m for m in list(MAIN_MODEL_ORDER) + [k for k in MODEL_METADATA if k not in MAIN_MODEL_ORDER]
        if MODEL_ENABLED.get(m, False) and MODEL_METADATA.get(m, {}).get("execution_scope") == "main_matrix"
    )
    plan = ExperimentPlan(
        registry=tuple(BUNDLE_REGISTRY),
        model_ids=active_main_models,
        model_seeds={k: tuple(v) for k, v in MODEL_SEEDS.items()},
        model_metadata=MODEL_METADATA,
        primary_metrics=PRIMARY_METRICS,
        reported_metrics=REPORTED_METRICS,
        class_metrics=CLASS_METRICS,
        metric_labels=METRIC_LABELS,
        reference_policy=REFERENCE_POLICY,
        graph_config=GRAPH_CFG,
        diagnostics=RELATIONAL_DIAGNOSTICS,
        special_experiments=tuple(x for x in SPECIAL_EXPERIMENTS if x.get("enabled", True)),
        registered_models=tuple(MODEL_METADATA),
    )
    if persist_registry:
        ensure_runtime_dirs()
        BUNDLE_TABLE.to_csv(STUDY_DIR / "bundle_registry.csv", index=False)
    return plan


def plan_summary(plan: ExperimentPlan, dataset: DatasetContext | None = None) -> dict[str, int]:
    """Compute counts of bundles, models, conditions, and total fold×seed model fits."""
    main_conditions = len(plan.registry) * len(plan.model_ids)
    special_conditions = len(plan.special_experiments)
    summary = {
        "bundles": len(plan.registry),
        "main_models": len(plan.model_ids),
        "main_conditions": main_conditions,
        "special_conditions": special_conditions,
        "conditions": main_conditions + special_conditions,
    }
    if dataset is not None:
        main_fits = sum(len(dataset.folds) * len(plan.model_seeds[m]) for m in plan.model_ids) * len(plan.registry)
        special_fits = sum(len(dataset.folds) * len(plan.model_seeds[e["model_id"]]) for e in plan.special_experiments)
        summary["fits"] = main_fits + special_fits
    return summary
