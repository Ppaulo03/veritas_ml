"""Pre-flight protocol and inductive integrity sanity checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.contracts import DatasetContext, ExperimentPlan, FeatureContext
from ..core.graph import build_knn_propagation, prepare_hetero_term_incidence
from ..features import (
    SOURCE_ORDER,
    WORD_CFGS,
    _fit_tfidf_base,
    bundle_component_time,
    combine_blocks,
)


def sanity_check_hetero_term(dataset: DatasetContext) -> dict[str, Any]:
    """Validate bipartite work-term incidence shapes and column-wise normalization."""
    tr, va = dataset.folds[0]
    texts_tr = dataset.frame.iloc[tr].texto.tolist()
    texts_va = dataset.frame.iloc[va].texto.tolist()
    _, a_tr, a_va, _, _ = _fit_tfidf_base(texts_tr, texts_va, WORD_CFGS["word_1"])
    rt, ct, re, meta = prepare_hetero_term_incidence(a_tr, a_va)

    assert rt.shape == (len(tr), a_tr.shape[1])
    assert ct.shape == rt.shape and re.shape == (len(va), a_tr.shape[1])

    nz = a_tr.toarray().sum(axis=0) > 1e-12
    if np.any(nz):
        assert np.allclose(ct[:, nz].sum(axis=0), 1.0, atol=1e-5)
    assert meta["term_nodes"] == a_tr.shape[1]
    return meta


def run_sanity_checks(
    dataset: DatasetContext,
    feature_context: FeatureContext,
    plan: ExperimentPlan,
) -> bool:
    """Verify fold disjointness, feature dimension stability, and inductive graph invariants."""
    for (tr, va), pack in zip(dataset.folds, feature_context.fold_packs):
        assert not (
            set(dataset.frame.iloc[tr].grupo_franquia)
            & set(dataset.frame.iloc[va].grupo_franquia)
        )
        for src in SOURCE_ORDER:
            assert pack["train"][src].shape[1] == pack["valid"][src].shape[1]

    tr, va = dataset.folds[0]
    xtr = combine_blocks(feature_context.fold_packs[0]["train"], ("minilm",))
    xva = combine_blocks(feature_context.fold_packs[0]["valid"], ("minilm",))
    p, pe, _ = build_knn_propagation(xtr, xva, **plan.graph_config)
    assert p.shape == (len(tr), len(tr)) and pe.shape == (len(tr) + len(va), len(tr) + len(va))
    assert not pe[: len(tr), len(tr) :].any()

    # Zero-signal regression check: sparse zero vectors must default to self-loops
    ztr = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    zva = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    zp, zpe, zedges = build_knn_propagation(ztr, zva, k=2, self_weight=0.9, min_similarity=None)
    assert np.count_nonzero(zp[1]) == 1 and zp[1, 1] == 1.0
    assert np.count_nonzero(zpe[len(ztr)]) == 1 and zpe[len(ztr), len(ztr)] == 1.0
    assert 1 not in set(zedges.origem_treino_local.astype(int))

    hetero_meta = sanity_check_hetero_term(dataset)
    assert hetero_meta["term_nodes"] > 0
    return True
