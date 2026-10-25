"""Compatibility shim re-exporting from media_gnn_study.engine.sanity."""

from .engine.sanity import (
    bundle_component_time,
    combine_blocks,
    run_sanity_checks,
    sanity_check_hetero_term,
)

__all__ = [
    "combine_blocks",
    "bundle_component_time",
    "run_sanity_checks",
    "sanity_check_hetero_term",
]
