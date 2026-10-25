"""Pure mathematical and inductive graph construction routines.

Guarantees:
- Strict inductive split: validation nodes receive messages exclusively from training nodes.
- Zero message passing from validation to training or validation to validation.
- Row normalization: propagation matrix rows always sum to 1.0.
- Zero-signal node isolation: works without active features in a fold default to self-loops.
- Unsupervised pruning: TF-IDF incidence reduction does not use target labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    import torch


def safe_unit_rows(x: np.ndarray | Any, eps: float = 1e-10) -> np.ndarray:
    """Normalize 2D array rows to unit L2 norm, preserving zero/near-zero vectors."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("Invalid feature block for row normalization.")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = arr.copy()
    valid_mask = norms[:, 0] >= eps
    out[valid_mask] /= norms[valid_mask]
    return out


def build_knn_propagation(
    x_train: np.ndarray,
    x_query: np.ndarray,
    k: int = 5,
    self_weight: float = 0.90,
    min_similarity: float | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build row-normalized inductive kNN propagation matrices.

    Parameters
    ----------
    x_train : np.ndarray
        Training feature matrix of shape (N, D).
    x_query : np.ndarray
        Validation or test query feature matrix of shape (M, D).
    k : int
        Number of nearest neighbors to select from training nodes.
    self_weight : float
        Weight assigned to self-loop (between 0.0 and 1.0).
    min_similarity : float, optional
        Cosine similarity threshold below which candidates are discarded.

    Returns
    -------
    p_train : np.ndarray
        Row-stochastic train-to-train transition matrix of shape (N, N).
    p_eval : np.ndarray
        Row-stochastic joint transition matrix of shape (N + M, N + M).
    edges_df : pd.DataFrame
        Edge list containing subset, destino_local, origem_treino_local, rank, cosseno.
    """
    xt = safe_unit_rows(x_train)
    xq = safe_unit_rows(x_query)
    n, m = len(xt), len(xq)

    if n < 2 or k < 1 or not (0.0 <= self_weight <= 1.0):
        raise ValueError("Invalid graph configuration parameters.")

    eps = 1e-10
    train_has_signal = np.linalg.norm(xt, axis=1) >= eps
    query_has_signal = np.linalg.norm(xq, axis=1) >= eps

    sim_train = xt @ xt.T
    sim_query = xq @ xt.T

    p_train = np.eye(n, dtype=np.float32)
    p_eval = np.zeros((n + m, n + m), dtype=np.float32)
    edges: list[tuple[str, int, int, int, float]] = []

    def _select_neighbors(similarity_row: np.ndarray, own_idx: int | None = None) -> np.ndarray:
        candidates = np.flatnonzero(train_has_signal)
        if own_idx is not None:
            candidates = candidates[candidates != own_idx]
        if len(candidates) == 0:
            return candidates

        ranked = candidates[np.argsort(-similarity_row[candidates], kind="stable")]
        ranked = ranked[: min(k, len(ranked))]
        if min_similarity is not None:
            ranked = ranked[similarity_row[ranked] >= min_similarity]
        return ranked

    # 1. Process training nodes (p_train and upper block of p_eval)
    for i in range(n):
        neighbors = _select_neighbors(sim_train[i], own_idx=i) if train_has_signal[i] else np.empty(0, dtype=int)
        p_train[i, :] = 0.0

        if len(neighbors) == 0:
            p_train[i, i] = 1.0
        else:
            p_train[i, i] = self_weight
            p_train[i, neighbors] = (1.0 - self_weight) / len(neighbors)

        for rank, j in enumerate(neighbors, start=1):
            edges.append(("treino", i, int(j), rank, float(sim_train[i, j])))

    p_eval[:n, :n] = p_train

    # 2. Process query / validation nodes (lower block of p_eval)
    for q in range(m):
        neighbors = _select_neighbors(sim_query[q]) if query_has_signal[q] else np.empty(0, dtype=int)
        row_idx = n + q

        if len(neighbors) == 0:
            p_eval[row_idx, row_idx] = 1.0
        else:
            p_eval[row_idx, row_idx] = self_weight
            p_eval[row_idx, neighbors] = (1.0 - self_weight) / len(neighbors)

        for rank, j in enumerate(neighbors, start=1):
            edges.append(("validacao", q, int(j), rank, float(sim_query[q, j])))

    # Assert inductive constraints and stochasticity
    assert np.allclose(p_train.sum(axis=1), 1.0), "p_train rows must sum to 1.0."
    assert np.allclose(p_eval.sum(axis=1), 1.0), "p_eval rows must sum to 1.0."
    assert not p_eval[:n, n:].any(), "No messages allowed from validation to training."

    # Assert zero-signal isolation
    zero_train = np.flatnonzero(~train_has_signal)
    if len(zero_train) > 0:
        assert np.allclose(p_train[zero_train, zero_train], 1.0)
        zero_set = set(zero_train.tolist())
        assert not any(int(j) in zero_set for _, _, j, _, _ in edges)

    zero_query = np.flatnonzero(~query_has_signal)
    for q in zero_query:
        row_idx = n + int(q)
        assert p_eval[row_idx, row_idx] == 1.0 and np.count_nonzero(p_eval[row_idx]) == 1

    edges_df = pd.DataFrame(
        edges,
        columns=["subset", "destino_local", "origem_treino_local", "rank", "cosseno"],
    )
    return p_train, p_eval, edges_df


def build_knn_edge_arrays(
    x_train: np.ndarray,
    x_query: np.ndarray,
    k: int = 5,
    min_similarity: float | None = None,
) -> np.ndarray:
    """Build directed edge index arrays (src, dst) as pure NumPy int64 array of shape (2, E).

    Inductive validation rule:
    Validation destination nodes only receive messages from training nodes.
    No messages originate from validation nodes.
    """
    _, _, edges_df = build_knn_propagation(
        x_train, x_query, k=k, self_weight=0.0, min_similarity=min_similarity
    )
    n = len(x_train)
    src: list[int] = []
    dst: list[int] = []

    for row in edges_df.itertuples(index=False):
        src.append(int(row.origem_treino_local))
        dest_idx = int(row.destino_local) if row.subset == "treino" else n + int(row.destino_local)
        dst.append(dest_idx)

    if not src:
        return np.empty((2, 0), dtype=np.int64)
    return np.array([src, dst], dtype=np.int64)


def build_knn_edge_index(
    x_train: np.ndarray,
    x_query: np.ndarray,
    k: int = 5,
    min_similarity: float | None = None,
    device: Any = None,
) -> torch.Tensor:
    """Convenience PyTorch tensor adapter for build_knn_edge_arrays."""
    import torch

    edge_arrays = build_knn_edge_arrays(x_train, x_query, k=k, min_similarity=min_similarity)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.as_tensor(edge_arrays, dtype=torch.long, device=target_device)


def prune_tfidf_topk(matrix: sparse.spmatrix, top_k: int | None = None) -> sparse.csr_matrix:
    """Retain only the top-k largest TF-IDF weights per row (work), completely unsupervised."""
    if top_k is None:
        return matrix.tocsr(copy=True)

    k = int(top_k)
    if k < 1:
        raise ValueError("top_k must be >= 1 or None.")

    csr = matrix.tocsr(copy=True)
    for i in range(csr.shape[0]):
        lo, hi = csr.indptr[i], csr.indptr[i + 1]
        if hi - lo <= k:
            continue
        vals = csr.data[lo:hi]
        keep = np.argpartition(vals, -k)[-k:]
        mask = np.ones(hi - lo, dtype=bool)
        mask[keep] = False
        vals[mask] = 0.0

    csr.eliminate_zeros()
    return csr


def prepare_hetero_term_incidence(
    a_train: sparse.spmatrix,
    a_eval: sparse.spmatrix,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute normalized work-term incidence matrices for bipartite heterogeneous GNN.

    Inductive policy:
    - Term states are computed exclusively from training works (col_normalized_train).
    - Validation works only query term states (row_normalized_eval); validation never updates terms.
    """
    pruned_train = prune_tfidf_topk(a_train, top_k)
    pruned_eval = prune_tfidf_topk(a_eval, top_k)

    train_arr = np.asarray(pruned_train.toarray(), dtype=np.float32)
    eval_arr = np.asarray(pruned_eval.toarray(), dtype=np.float32)

    # Column normalization (work -> term): only training works define term states
    col_sum = train_arr.sum(axis=0, keepdims=True)
    col_norm_train = np.divide(
        train_arr, col_sum, out=np.zeros_like(train_arr), where=col_sum > 1e-12
    )

    # Row normalization (term -> work): TF-IDF weights normalized per work
    row_sum_tr = train_arr.sum(axis=1, keepdims=True)
    row_sum_ev = eval_arr.sum(axis=1, keepdims=True)
    row_norm_train = np.divide(
        train_arr, row_sum_tr, out=np.zeros_like(train_arr), where=row_sum_tr > 1e-12
    )
    row_norm_eval = np.divide(
        eval_arr, row_sum_ev, out=np.zeros_like(eval_arr), where=row_sum_ev > 1e-12
    )

    metadata = {
        "term_nodes": int(train_arr.shape[1]),
        "incidence_train": int(np.count_nonzero(train_arr)),
        "incidence_eval": int(np.count_nonzero(eval_arr)),
        "zero_term_train_rows": int(np.sum(row_sum_tr[:, 0] <= 1e-12)),
        "zero_term_eval_rows": int(np.sum(row_sum_ev[:, 0] <= 1e-12)),
        "term_top_k": None if top_k is None else int(top_k),
    }
    return row_norm_train, col_norm_train, row_norm_eval, metadata
