"""Model-agnostic registry for notebook-defined classifiers and special experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.config import DEFAULT_MODEL_SEEDS
from ..core.contracts import GraphContext
from .base import ModelAdapterFn, ModelOutput, ModelSpec

_MODEL_SPECS: dict[str, ModelSpec] = {}
_SPECIAL_EXPERIMENTS: list[dict[str, Any]] = []


def register_model(
    model_id: str,
    adapter_fn: ModelAdapterFn,
    *,
    display_name: str | None = None,
    family: str = "custom",
    role: str = "individual",
    seeds: tuple[int, ...] = DEFAULT_MODEL_SEEDS,
    enabled: bool = True,
    execution_scope: str = "main_matrix",
    metadata: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
) -> ModelSpec:
    """Register or override a model adapter in the experimental framework.

    Designed to be called from notebooks to inject PyTorch, TensorFlow/Keras,
    JAX, or scikit-learn/XGBoost models without modifying the core framework.
    """
    meta = dict(metadata or {})
    resolved_display = str(meta.get("display_name", display_name or model_id))
    resolved_family = str(meta.get("family", family))
    resolved_role = str(meta.get("role", role))
    resolved_enabled = bool(meta.get("enabled", enabled))
    resolved_scope = str(meta.get("execution_scope", execution_scope))
    resolved_train_cfg = dict(training_config or meta.get("training_config") or {})

    spec = ModelSpec(
        model_id=model_id,
        adapter_fn=adapter_fn,
        display_name=resolved_display,
        family=resolved_family,
        role=resolved_role,
        seeds=tuple(int(s) for s in seeds),
        enabled=resolved_enabled,
        execution_scope=resolved_scope,
        training_config=resolved_train_cfg,
    )
    _MODEL_SPECS[model_id] = spec
    return spec


def register_special_experiment(spec: dict[str, Any]) -> None:
    """Register a special experiment specification (e.g., operator comparison or hetero graph).

    Replaces any existing specification with the same ``experiment_id``.
    """
    exp_id = str(spec.get("experiment_id", f"special_{len(_SPECIAL_EXPERIMENTS)}"))
    normalized = dict(spec)
    normalized["experiment_id"] = exp_id
    for idx, existing in enumerate(_SPECIAL_EXPERIMENTS):
        if existing.get("experiment_id") == exp_id:
            _SPECIAL_EXPERIMENTS[idx] = normalized
            return
    _SPECIAL_EXPERIMENTS.append(normalized)


def clear_registry() -> None:
    """Remove all registered models and special experiments."""
    _MODEL_SPECS.clear()
    _SPECIAL_EXPERIMENTS.clear()


def get_model(model_id: str) -> ModelAdapterFn:
    """Retrieve a registered model adapter function by its identifier."""
    if model_id not in _MODEL_SPECS:
        raise KeyError(
            f"Model '{model_id}' is not registered. "
            f"Available models: {list(_MODEL_SPECS.keys())}"
        )
    return _MODEL_SPECS[model_id].adapter_fn


def get_model_spec(model_id: str) -> ModelSpec:
    """Retrieve the full ModelSpec for a registered model."""
    if model_id not in _MODEL_SPECS:
        raise KeyError(
            f"Model '{model_id}' is not registered. "
            f"Available models: {list(_MODEL_SPECS.keys())}"
        )
    return _MODEL_SPECS[model_id]


def list_models(
    *,
    enabled_only: bool = False,
    execution_scope: str | None = None,
) -> list[str]:
    """List registered model identifiers in registration order."""
    out: list[str] = []
    for model_id, spec in _MODEL_SPECS.items():
        if enabled_only and not spec.enabled:
            continue
        if execution_scope is not None and spec.execution_scope != execution_scope:
            continue
        out.append(model_id)
    return out


def get_model_metadata() -> dict[str, dict[str, Any]]:
    """Return metadata dictionaries for all registered models."""
    return {
        model_id: spec.to_metadata_dict()
        for model_id, spec in _MODEL_SPECS.items()
    }


def get_model_seeds() -> dict[str, tuple[int, ...]]:
    """Return configured seed tuples for all registered models."""
    return {
        model_id: spec.seeds
        for model_id, spec in _MODEL_SPECS.items()
    }


def get_model_enabled() -> dict[str, bool]:
    """Return enabled flags for all registered models."""
    return {
        model_id: spec.enabled
        for model_id, spec in _MODEL_SPECS.items()
    }


def get_special_experiments() -> list[dict[str, Any]]:
    """Return copies of all registered special experiment definitions."""
    return [dict(item) for item in _SPECIAL_EXPERIMENTS]


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
