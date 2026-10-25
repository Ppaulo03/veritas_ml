"""Base abstractions, protocols, and return contracts for models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

import numpy as np

from ..core.contracts import GraphContext


@dataclass(frozen=True)
class ModelOutput:
    """Standardized output returned by any classifier backend.

    Supports tuple unpacking:
    prob, complexity, kind, train_s, pred_s = output
    """

    probabilities: np.ndarray
    complexity: int
    complexity_type: str
    train_seconds: float
    predict_seconds: float

    def __iter__(self) -> Iterator:
        return iter((
            self.probabilities,
            self.complexity,
            self.complexity_type,
            self.train_seconds,
            self.predict_seconds,
        ))


class ModelAdapterFn(Protocol):
    """Universal signature for any model adapter (PyTorch, TensorFlow, Classical, etc.)."""

    def __call__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        graph_context: GraphContext | None,
        seed: int,
    ) -> ModelOutput | tuple[np.ndarray, int, str, float, float]:
        ...


@dataclass(frozen=True)
class ModelSpec:
    """Registration metadata and adapter binding for a notebook-defined model."""

    model_id: str
    adapter_fn: ModelAdapterFn
    display_name: str
    family: str = "custom"
    role: str = "individual"
    seeds: tuple[int, ...] = (0, 1, 2)
    enabled: bool = True
    execution_scope: str = "main_matrix"
    training_config: dict[str, Any] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize model descriptor to a dictionary for manifests and reports."""
        meta: dict[str, Any] = {
            "display_name": self.display_name,
            "family": self.family,
            "role": self.role,
            "enabled": self.enabled,
            "execution_scope": self.execution_scope,
            "seeds": list(self.seeds),
        }
        if self.training_config:
            meta["training_config"] = dict(self.training_config)
        return meta
