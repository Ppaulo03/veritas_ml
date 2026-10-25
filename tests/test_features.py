"""Tests for lexical feature extraction, SVD reduction, and block combination."""

import numpy as np
from veritas_ml.features import (
    WORD_CFGS,
    _fit_svd,
    _fit_tfidf_base,
    bundle_component_time,
    combine_blocks,
)


def test_combine_blocks_l2_normalizes_concatenation():
    blocks = {
        "minilm": np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        "word_1": np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32),
    }
    combined = combine_blocks(blocks, ("minilm", "word_1"))
    assert combined.shape == (2, 4)
    norms = np.linalg.norm(combined, axis=1)
    assert np.allclose(norms, 1.0)


def test_bundle_component_time_deduplicates_shared_base():
    pack = {
        "components": {
            "word_1_2_base": {"fit_seconds": 1.5, "transform_seconds": 0.5},
            "word_1_2_svd": {"fit_seconds": 0.8, "transform_seconds": 0.2},
            "topics_nmf": {"fit_seconds": 1.2, "transform_seconds": 0.3},
        }
    }
    fit_s, trans_s, unique = bundle_component_time(pack, ("word_1_2", "topics"))
    assert unique == ["word_1_2_base", "word_1_2_svd", "topics_nmf"]
    assert np.isclose(fit_s, 1.5 + 0.8 + 1.2)
    assert np.isclose(trans_s, 0.5 + 0.2 + 0.3)


def test_fit_tfidf_and_svd_on_synthetic_texts():
    texts_tr = [
        "guerra espacial nave frota planeta batalha",
        "guerra espacial batalha estrela comandante",
        "romance escola amor amizade casal sentimento",
        "romance escola paixão amizade estudante",
        "terror monstro sombra noite susto pesadelo",
        "terror monstro escuridão medo pesadelo",
    ]
    texts_va = [
        "guerra nave comandante planeta",
        "romance amor estudante escola",
    ]
    _, a_tr, a_va, _, _ = _fit_tfidf_base(texts_tr, texts_va, WORD_CFGS["word_1"])
    xtr, xva, _, _, _ = _fit_svd(a_tr, a_va, target_dim=3)
    assert xtr.shape[0] == len(texts_tr)
    assert xva.shape[0] == len(texts_va)
    assert xtr.shape[1] == xva.shape[1]
