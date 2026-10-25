"""Evaluation metrics for multilabel classification."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from ..core.config import FIXED_THRESHOLD


def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, float]:
    """Compute micro- and macro-averaged F1, Precision, and Recall."""
    pred = np.asarray(y_prob) >= threshold
    return {
        "f1_micro": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, pred, average="micro", zero_division=0)),
    }


def class_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: Sequence[str],
    threshold: float = FIXED_THRESHOLD,
) -> pd.DataFrame:
    """Compute per-class Precision, Recall, F1, and Support."""
    pred = np.asarray(y_prob) >= threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, average=None, zero_division=0
    )
    return pd.DataFrame({
        "class_name": list(classes),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
