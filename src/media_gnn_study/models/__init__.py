"""Model contracts, evaluation metrics, and pluggable adapter registry."""

from __future__ import annotations

from ..core.graph import (
    build_knn_edge_arrays,
    build_knn_edge_index,
    build_knn_propagation,
    prepare_hetero_term_incidence,
    prune_tfidf_topk,
)
from .base import ModelAdapterFn, ModelOutput, ModelSpec
from .metrics import class_metrics, multilabel_metrics
from .registry import (
    clear_registry,
    get_model,
    get_model_enabled,
    get_model_metadata,
    get_model_seeds,
    get_model_spec,
    get_special_experiments,
    list_models,
    register_model,
    register_special_experiment,
    run_model,
)

__all__ = [
    "ModelOutput",
    "ModelAdapterFn",
    "ModelSpec",
    "register_model",
    "register_special_experiment",
    "clear_registry",
    "get_model",
    "get_model_spec",
    "get_model_metadata",
    "get_model_seeds",
    "get_model_enabled",
    "get_special_experiments",
    "list_models",
    "run_model",
    "multilabel_metrics",
    "class_metrics",
    "build_knn_propagation",
    "build_knn_edge_arrays",
    "build_knn_edge_index",
    "prune_tfidf_topk",
    "prepare_hetero_term_incidence",
]
