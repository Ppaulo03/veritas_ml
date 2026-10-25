"""Tests for core contracts and pure numpy graph routines."""

import numpy as np
from media_gnn_study.core.contracts import stable_id
from media_gnn_study.core.graph import build_knn_edge_arrays, safe_unit_rows


def test_stable_id_deterministic():
    id1 = stable_id("cond", "minilm+word_1|gnn")
    id2 = stable_id("cond", "minilm+word_1|gnn")
    assert id1 == id2
    assert id1.startswith("cond_")
    assert len(id1) == 5 + 12


def test_safe_unit_rows_preserves_zero():
    x = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    normed = safe_unit_rows(x)
    assert np.allclose(normed[0], [0.0, 0.0])
    assert np.allclose(normed[1], [0.6, 0.8])


def test_build_knn_edge_arrays_returns_numpy_shape():
    tr = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    va = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    edges = build_knn_edge_arrays(tr, va, k=2)
    assert isinstance(edges, np.ndarray)
    assert edges.shape[0] == 2
    assert edges.dtype == np.int64
