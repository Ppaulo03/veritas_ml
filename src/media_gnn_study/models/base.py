"""Base abstractions, protocols, and return contracts for models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

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
