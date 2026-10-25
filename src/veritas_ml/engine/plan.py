"""Experimental bundle catalog and model-agnostic ExperimentPlan assembly."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from ..core.config import (
    CLASS_METRICS,
    GRAPH_CFG,
    METRIC_LABELS,
    PRIMARY_METRICS,
    REFERENCE_POLICY,
    RELATIONAL_DIAGNOSTICS,
    REPORTED_METRICS,
    STUDY_DIR,
    ensure_runtime_dirs,
)
from ..core.contracts import DatasetContext, ExperimentPlan
from ..features import SOURCE_ORDER
from ..models.registry import (
    get_model_metadata,
    get_model_seeds,
    get_special_experiments,
    list_models,
)


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


def build_experiment_plan(
    dataset: DatasetContext | None = None,
    *,
    persist_registry: bool = True,
    registry: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    model_ids: tuple[str, ...] | list[str] | None = None,
    special_experiments: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> ExperimentPlan:
    """Assemble ExperimentPlan dynamically from the notebook-populated model registry."""
    meta = get_model_metadata()
    seeds = get_model_seeds()

    if model_ids is not None:
        active_main_models = tuple(str(m) for m in model_ids)
    else:
        active_main_models = tuple(
            list_models(enabled_only=True, execution_scope="main_matrix")
        )

    raw_specials = (
        list(special_experiments)
        if special_experiments is not None
        else get_special_experiments()
    )
    active_specials = tuple(
        x for x in raw_specials
        if x.get("enabled", True) and x.get("model_id") in meta
    )

    active_registry = tuple(registry) if registry is not None else tuple(BUNDLE_REGISTRY)

    plan = ExperimentPlan(
        registry=active_registry,
        model_ids=active_main_models,
        model_seeds=seeds,
        model_metadata=meta,
        primary_metrics=PRIMARY_METRICS,
        reported_metrics=REPORTED_METRICS,
        class_metrics=CLASS_METRICS,
        metric_labels=METRIC_LABELS,
        reference_policy=REFERENCE_POLICY,
        graph_config=GRAPH_CFG,
        diagnostics=RELATIONAL_DIAGNOSTICS,
        special_experiments=active_specials,
        registered_models=tuple(meta.keys()),
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
        main_fits = (
            sum(len(dataset.folds) * len(plan.model_seeds[m]) for m in plan.model_ids)
            * len(plan.registry)
        )
        special_fits = sum(
            len(dataset.folds) * len(plan.model_seeds[e["model_id"]])
            for e in plan.special_experiments
        )
        summary["fits"] = main_fits + special_fits
    return summary
