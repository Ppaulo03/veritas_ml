"""Experimental plan, sanity checks, and framework-agnostic execution runner."""

from .plan import (
    BUNDLE_REGISTRY,
    BUNDLE_TABLE,
    CORE_SOURCES,
    SPECIAL_EXPERIMENTS,
    build_experiment_plan,
    plan_summary,
)
from .runner import (
    execute_experiments,
    load_incremental_base,
    persist_experiment_result,
    standardize_experiment_result,
)
from .sanity import (
    bundle_component_time,
    combine_blocks,
    run_sanity_checks,
    sanity_check_hetero_term,
)

__all__ = [
    "CORE_SOURCES",
    "BUNDLE_REGISTRY",
    "BUNDLE_TABLE",
    "SPECIAL_EXPERIMENTS",
    "build_experiment_plan",
    "plan_summary",
    "combine_blocks",
    "bundle_component_time",
    "run_sanity_checks",
    "sanity_check_hetero_term",
    "standardize_experiment_result",
    "persist_experiment_result",
    "load_incremental_base",
    "execute_experiments",
]
