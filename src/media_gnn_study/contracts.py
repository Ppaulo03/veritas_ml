"""Compatibility shim re-exporting from media_gnn_study.core.contracts."""

from .core.contracts import (
    DatasetContext,
    FeatureContext,
    ExperimentPlan,
    ExperimentResult,
    StudyResult,
    stable_id,
)

__all__ = [
    "DatasetContext",
    "FeatureContext",
    "ExperimentPlan",
    "ExperimentResult",
    "StudyResult",
    "stable_id",
]
