"""Model contracts, evaluation metrics, and pluggable adapter registry."""

from __future__ import annotations

from typing import Any

from ..core.graph import (
    build_knn_edge_arrays,
    build_knn_edge_index,
    build_knn_propagation,
    prepare_hetero_term_incidence,
    prune_tfidf_topk,
)
from .backends.classical import RF_CFG, XGB_CFG, fit_predict_classical
from .base import ModelAdapterFn, ModelOutput
from .metrics import class_metrics, multilabel_metrics
from .registry import (
    get_model,
    list_models,
    register_builtin_models,
    register_model,
    run_model,
)

__all__ = [
    "ModelOutput",
    "ModelAdapterFn",
    "register_model",
    "get_model",
    "list_models",
    "run_model",
    "register_builtin_models",
    "multilabel_metrics",
    "class_metrics",
    "build_knn_propagation",
    "build_knn_edge_arrays",
    "build_knn_edge_index",
    "prune_tfidf_topk",
    "prepare_hetero_term_incidence",
    "RF_CFG",
    "XGB_CFG",
    "fit_predict_classical",
    "fit_predict_neural",
    "fit_predict_homogeneous_operator",
    "fit_predict_hetero_term",
    "fit_predict_hetero_term_gatv2",
    "LinearClassifier",
    "MLPGraphClassifier",
    "HomogeneousPyGClassifier",
    "HeteroTermGNN",
    "HeteroTermGATv2",
]

_TORCH_EXPORTS = {
    "LinearClassifier",
    "MLPGraphClassifier",
    "HomogeneousPyGClassifier",
    "HeteroTermGNN",
    "HeteroTermGATv2",
    "fit_predict_neural",
    "fit_predict_homogeneous_operator",
    "fit_predict_hetero_term",
    "fit_predict_hetero_term_gatv2",
}


def __getattr__(name: str) -> Any:
    """Lazy load PyTorch backend symbols only when explicitly requested."""
    if name in _TORCH_EXPORTS:
        from .backends import torch as torch_backend

        return getattr(torch_backend, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
