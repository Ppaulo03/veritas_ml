"""Compatibility shim re-exporting from media_gnn_study.engine.plan."""

from .engine.plan import (
    BUNDLE_REGISTRY,
    BUNDLE_TABLE,
    CORE_SOURCES,
    SPECIAL_EXPERIMENTS,
    build_experiment_plan,
    plan_summary,
)

__all__ = [
    "CORE_SOURCES",
    "BUNDLE_REGISTRY",
    "BUNDLE_TABLE",
    "SPECIAL_EXPERIMENTS",
    "build_experiment_plan",
    "plan_summary",
]
