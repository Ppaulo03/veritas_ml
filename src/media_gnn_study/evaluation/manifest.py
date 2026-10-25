"""Reproducibility manifest generation and persistence."""

from __future__ import annotations

import json
from typing import Any

from ..core.config import (
    CV_SEED,
    DROPOUT,
    EPOCHS,
    FIXED_THRESHOLD,
    HETERO_DIFFUSION_CFG,
    HETERO_TERM_CFG,
    HIDDEN,
    HOMOGENEOUS_OPERATOR_CFG,
    INCREMENTAL_BASE_RUN_ID,
    INCREMENTAL_REUSE,
    LR,
    MODEL_METADATA,
    RELATIONAL_RESEARCH_REGISTRY,
    RUN_DIR,
    RUN_ID,
    RUNTIME_INFO,
    STUDY_DIR,
    STUDY_VERSION,
    WEIGHT_DECAY,
    ensure_runtime_dirs,
)
from ..core.contracts import (
    DatasetContext,
    ExperimentPlan,
    ExperimentResult,
    FeatureContext,
    StudyResult,
)
from .diagnostics import GRAPH_VIS_CONFIG, LEARNING_CURVE_CONFIG


def write_manifest(
    dataset: DatasetContext,
    feature_context: FeatureContext,
    experiment_result: ExperimentResult,
    study_result: StudyResult,
    diagnostics: dict[str, Any],
    plan: ExperimentPlan,
) -> dict[str, Any]:
    """Write reproducibility manifest JSON to STUDY_DIR and RUN_DIR."""
    manifest = {
        "status": "COMPLETE_DECOUPLED_STUDY_WITH_GRAPH_OPERATOR_AND_HETERO_DIFFUSION_ABLATIONS",
        "study_version": STUDY_VERSION,
        "run_id": RUN_ID,
        "architecture": {
            "dataset_contract": "DatasetContext",
            "experiment_contract": experiment_result.schema_version,
            "presentation_contract": study_result.schema_version,
            "renderer_entrypoint": "render_study(StudyResult)",
        },
        "dataset": dataset.metadata,
        "audit_used": False,
        "classes": list(dataset.classes),
        "cv": {
            "outer_splits": len(dataset.folds),
            "group": "grupo_franquia",
            "fold_seed": CV_SEED,
            "model_seeds": {m: list(s) for m, s in plan.model_seeds.items()},
        },
        "device": "cpu",
        "runtime": RUNTIME_INFO,
        "training": {
            "epochs": EPOCHS,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "threshold": FIXED_THRESHOLD,
            "loss": "BCEWithLogitsLoss_pos_weight_train_only",
        },
        "graph": plan.graph_config,
        "models": plan.model_metadata,
        "model_execution_enabled": {m: bool(v.get("enabled", True)) for m, v in MODEL_METADATA.items()},
        "special_experiments": [dict(x) for x in plan.special_experiments],
        "homogeneous_operator_config": HOMOGENEOUS_OPERATOR_CFG,
        "hetero_term_config": HETERO_TERM_CFG,
        "hetero_diffusion_config": HETERO_DIFFUSION_CFG,
        "incremental_reuse": {"enabled": INCREMENTAL_REUSE, "base_run_id": INCREMENTAL_BASE_RUN_ID},
        "relational_research_registry": RELATIONAL_RESEARCH_REGISTRY,
        "feature_extractors": feature_context.fixed_meta,
        "n_bundles": len(plan.registry),
        "n_conditions": len(experiment_result.conditions),
        "primary_metrics": list(plan.primary_metrics),
        "reference_policy": plan.reference_policy,
        "reference": study_result.reference,
        "analysis_version": "agnostic_renderer_v5",
        "diagnostics": sorted(diagnostics.keys()),
        "learning_curve": LEARNING_CURVE_CONFIG,
        "graph_visualization": GRAPH_VIS_CONFIG,
        "target_labels_used_to_construct_features_or_edges": False,
        "labels_used_posthoc_for_diagnostics_only": True,
        "graph_visualization_labels_posthoc_only": True,
        "learning_curve_subset_selection_uses_outer_train_labels_only": True,
        "multiple_comparisons_note": "exploratory development screening; reference is descriptive, not final unbiased evidence",
        "generative_calls": 0,
    }
    ensure_runtime_dirs()
    for p in (STUDY_DIR / "manifesto.json", RUN_DIR / "manifesto.json"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest
