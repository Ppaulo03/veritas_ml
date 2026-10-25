"""Pluggable model registry supporting notebook-defined and built-in adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.config import MODEL_ENABLED, MODEL_METADATA, MODEL_SEEDS, NEURAL_SEEDS
from ..core.contracts import GraphContext
from .base import ModelAdapterFn, ModelOutput

_MODEL_REGISTRY: dict[str, ModelAdapterFn] = {}


def register_model(
    model_id: str,
    adapter_fn: ModelAdapterFn,
    metadata: dict[str, Any] | None = None,
    seeds: tuple[int, ...] = NEURAL_SEEDS,
    enabled: bool = True,
) -> None:
    """Register or override a model adapter in the experimental framework.

    Can be called from a notebook to inject TensorFlow, Keras, JAX, or custom PyTorch models.
    """
    _MODEL_REGISTRY[model_id] = adapter_fn
    MODEL_ENABLED[model_id] = enabled
    if model_id not in MODEL_SEEDS:
        MODEL_SEEDS[model_id] = seeds
    if metadata is not None:
        base_meta = {
            "display_name": model_id,
            "family": "custom",
            "role": "custom",
            "enabled": enabled,
            "execution_scope": "main_matrix",
        }
        base_meta.update(metadata)
        MODEL_METADATA[model_id] = base_meta


def get_model(model_id: str) -> ModelAdapterFn:
    """Retrieve a registered model adapter by its identifier."""
    if model_id not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{model_id}' is not registered. Available models: {sorted(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_id]


def list_models() -> list[str]:
    """List all currently registered model identifiers."""
    return sorted(_MODEL_REGISTRY.keys())


def run_model(
    model_id: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    graph_context: GraphContext | None,
    seed: int,
) -> ModelOutput:
    """Execute a registered model and guarantee a standardized ModelOutput return."""
    adapter = get_model(model_id)
    result = adapter(x_train, y_train, x_eval, graph_context, seed)
    if isinstance(result, ModelOutput):
        return result
    prob, complexity, kind, train_s, pred_s = result
    return ModelOutput(
        probabilities=np.asarray(prob, dtype=np.float32),
        complexity=int(complexity),
        complexity_type=str(kind),
        train_seconds=float(train_s),
        predict_seconds=float(pred_s),
    )


# ---------------- Built-in Lazy Adapters ----------------

def _make_neural_adapter(model_name: str) -> ModelAdapterFn:
    def _adapter(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        graph_context: GraphContext | None,
        seed: int,
    ) -> ModelOutput:
        from .backends.torch.trainer import fit_predict_neural

        p_tr = graph_context.p_train if graph_context is not None else None
        p_ev = graph_context.p_eval if graph_context is not None else None
        return fit_predict_neural(x_train, y_train, x_eval, model_name, seed, p_tr, p_ev)

    return _adapter


def _make_classical_adapter(model_name: str) -> ModelAdapterFn:
    def _adapter(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        graph_context: GraphContext | None,
        seed: int,
    ) -> ModelOutput:
        from .backends.classical import fit_predict_classical

        return fit_predict_classical(x_train, y_train, x_eval, model_name, seed)

    return _adapter


def _make_pyg_operator_adapter(operator_name: str) -> ModelAdapterFn:
    def _adapter(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        graph_context: GraphContext | None,
        seed: int,
    ) -> ModelOutput:
        import torch
        from ..core.config import DEVICE
        from .backends.torch.trainer import fit_predict_homogeneous_operator

        if graph_context is None or graph_context.edge_index_train is None or graph_context.edge_index_eval is None:
            raise ValueError(f"Model '{operator_name}' requires edge_index_train and edge_index_eval in GraphContext.")

        edge_tr = torch.as_tensor(graph_context.edge_index_train, dtype=torch.long, device=DEVICE)
        edge_ev = torch.as_tensor(graph_context.edge_index_eval, dtype=torch.long, device=DEVICE)
        return fit_predict_homogeneous_operator(
            x_train, y_train, x_eval, operator_name, seed, edge_tr, edge_ev
        )

    return _adapter


def _hetero_term_gnn_adapter(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    graph_context: GraphContext | None,
    seed: int,
) -> ModelOutput:
    from .backends.torch.trainer import fit_predict_hetero_term

    if (
        graph_context is None
        or graph_context.row_norm_train is None
        or graph_context.col_norm_train is None
        or graph_context.row_norm_eval is None
    ):
        raise ValueError("HeteroTermGNN requires bipartite incidence matrices in GraphContext.")

    return fit_predict_hetero_term(
        x_train,
        y_train,
        x_eval,
        graph_context.row_norm_train,
        graph_context.col_norm_train,
        graph_context.row_norm_eval,
        seed,
    )


def _hetero_term_gatv2_adapter(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    graph_context: GraphContext | None,
    seed: int,
) -> ModelOutput:
    from .backends.torch.trainer import fit_predict_hetero_term_gatv2

    if (
        graph_context is None
        or graph_context.row_norm_train is None
        or graph_context.col_norm_train is None
        or graph_context.row_norm_eval is None
    ):
        raise ValueError("HeteroTermGATv2 requires bipartite incidence matrices in GraphContext.")

    return fit_predict_hetero_term_gatv2(
        x_train,
        y_train,
        x_eval,
        graph_context.row_norm_train,
        graph_context.col_norm_train,
        graph_context.row_norm_eval,
        seed,
    )


def register_builtin_models() -> None:
    """Register the default reference models."""
    for name in ("linear", "mlp", "gnn"):
        _MODEL_REGISTRY[name] = _make_neural_adapter(name)
    for name in ("rf", "xgb"):
        _MODEL_REGISTRY[name] = _make_classical_adapter(name)
    for name in ("gcn", "graphsage", "gatv2"):
        _MODEL_REGISTRY[name] = _make_pyg_operator_adapter(name)
    _MODEL_REGISTRY["hetero_term_gnn"] = _hetero_term_gnn_adapter
    _MODEL_REGISTRY["hetero_term_gatv2"] = _hetero_term_gatv2_adapter


register_builtin_models()
