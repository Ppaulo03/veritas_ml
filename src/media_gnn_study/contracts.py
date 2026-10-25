"""Compatibility shim re-exporting from media_gnn_study.core.contracts."""

from .core.contracts import (
    DatasetContext,
    FeatureContext,
    GraphContext,
    ExperimentPlan,
    ExperimentResult,
    StudyResult,
    stable_id,
)

__all__ = [
    "DatasetContext",
    "FeatureContext",
    "GraphContext",
    "ExperimentPlan",
    "ExperimentResult",
    "StudyResult",
    "stable_id",
]
