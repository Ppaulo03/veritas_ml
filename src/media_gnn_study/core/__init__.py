"""Core primitives, configuration, contracts, and graph mathematics."""

from .config import *
from .contracts import (
    DatasetContext,
    ExperimentPlan,
    ExperimentResult,
    FeatureContext,
    GraphContext,
    StudyResult,
    stable_id,
)
from .graph import (
    build_knn_edge_arrays,
    build_knn_edge_index,
    build_knn_propagation,
    prepare_hetero_term_incidence,
    prune_tfidf_topk,
    safe_unit_rows,
)
