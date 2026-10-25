"""Diagnostics, StudyResult assembly, rendering, and reproducibility manifests."""

from .diagnostics import (
    GRAPH_VIS_CONFIG,
    LEARNING_CURVE_CONFIG,
    run_diagnostics,
    run_graph_operator_diagnostic,
    run_graph_visualization_diagnostic,
    run_hetero_structural_diagnostic,
    run_learning_curve_diagnostic,
    run_relational_diagnostics,
)
from .manifest import write_manifest
from .render import persist_study_result, render_study, validate_study_result
from .study import (
    assemble_study_result,
    build_dataset_profile,
    build_model_overview,
    build_plot_specs,
    build_reference_by_class,
    pareto_front,
    select_reference,
    summarize_experiment,
)

__all__ = [
    "LEARNING_CURVE_CONFIG",
    "GRAPH_VIS_CONFIG",
    "run_relational_diagnostics",
    "run_learning_curve_diagnostic",
    "run_graph_visualization_diagnostic",
    "run_graph_operator_diagnostic",
    "run_hetero_structural_diagnostic",
    "run_diagnostics",
    "build_dataset_profile",
    "summarize_experiment",
    "build_model_overview",
    "select_reference",
    "build_reference_by_class",
    "pareto_front",
    "build_plot_specs",
    "assemble_study_result",
    "validate_study_result",
    "persist_study_result",
    "render_study",
    "write_manifest",
]
