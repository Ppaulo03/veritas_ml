"""Core data contracts for media GNN study.

Defines boundary dataclasses between pipeline stages:
- DatasetContext: frozen dataset and fold split state
- FeatureContext: extracted representations per fold
- ExperimentPlan: catalog of bundles and model conditions
- ExperimentResult: raw and standardized evaluation observations
- StudyResult: final structured tables and render specs
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def stable_id(prefix: str, value: Any) -> str:
    """Generate a deterministic 12-char hex identifier from any serializable value."""
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class GraphContext:
    """Inductive graph topology data delivered to models (100% framework-agnostic in NumPy)."""

    p_train: np.ndarray | None = None
    p_eval: np.ndarray | None = None
    edge_index_train: np.ndarray | None = None
    edge_index_eval: np.ndarray | None = None
    edges_df: pd.DataFrame | None = None
    row_norm_train: np.ndarray | None = None
    col_norm_train: np.ndarray | None = None
    row_norm_eval: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetContext:
    """Immutable snapshot of the dataset, target labels, and CV partition structure."""

    frame: pd.DataFrame
    targets: np.ndarray
    classes: tuple[str, ...]
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    fold_id: np.ndarray
    partitions: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureContext:
    """Extracted feature representations, metadata, and timing profiles per fold."""

    fold_packs: list[dict[str, Any]]
    catalog: pd.DataFrame
    fixed_meta: dict[str, Any]
    fixed_extraction_timing: pd.DataFrame
    fold_meta: pd.DataFrame
    component_timing: pd.DataFrame


@dataclass(frozen=True)
class ExperimentPlan:
    """Specification of all conditions, models, seeds, and evaluation metrics."""

    registry: tuple[dict[str, Any], ...]
    model_ids: tuple[str, ...]
    model_seeds: dict[str, tuple[int, ...]]
    model_metadata: dict[str, dict[str, Any]]
    primary_metrics: tuple[str, ...]
    reported_metrics: tuple[str, ...]
    class_metrics: tuple[str, ...]
    metric_labels: dict[str, str]
    reference_policy: dict[str, Any]
    graph_config: dict[str, Any]
    diagnostics: tuple[dict[str, Any], ...] = ()
    special_experiments: tuple[dict[str, Any], ...] = ()
    registered_models: tuple[str, ...] = ()


@dataclass
class ExperimentResult:
    """Structured observations and metrics collected from executing an experiment plan."""

    schema_version: str
    conditions: pd.DataFrame
    metric_observations: pd.DataFrame
    class_metric_observations: pd.DataFrame
    timing_observations: pd.DataFrame
    raw_metrics: pd.DataFrame
    raw_class_metrics: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StudyResult:
    """Final assembled study artifacts, tables, and visualization specifications."""

    schema_version: str
    metadata: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    reference: dict[str, Any]
    table_specs: list[dict[str, Any]]
    plot_specs: list[dict[str, Any]]
