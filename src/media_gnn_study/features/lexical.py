"""Lexical TF-IDF/SVD/NMF feature extraction and fold-local FeatureContext assembly."""

from __future__ import annotations

import time
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from ..core.config import (
    ACTION_SVD_DIM,
    CV_SEED,
    LEX_SVD_DIM,
    NMF_DIM,
    STUDY_DIR,
)
from ..core.contracts import DatasetContext, FeatureContext
from ..core.graph import safe_unit_rows
from .encoders import get_emotion_features, get_minilm_features
from .linguistic import get_spacy_text_views

BASE_TFIDF_CFG: dict[str, Any] = {
    "lowercase": True,
    "strip_accents": "unicode",
    "analyzer": "word",
    "ngram_range": (1, 2),
    "token_pattern": r"(?u)\b\w\w+\b",
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 20000,
    "sublinear_tf": True,
    "smooth_idf": True,
    "norm": "l2",
    "dtype": np.float64,
}

WORD_CFGS: dict[str, dict[str, Any]] = {
    "word_1": {**BASE_TFIDF_CFG, "ngram_range": (1, 1)},
    "word_1_2": {**BASE_TFIDF_CFG, "ngram_range": (1, 2)},
    "word_1_3": {**BASE_TFIDF_CFG, "ngram_range": (1, 3)},
}

CHAR_CFG: dict[str, Any] = {
    "lowercase": True,
    "strip_accents": "unicode",
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 30000,
    "sublinear_tf": True,
    "smooth_idf": True,
    "norm": "l2",
    "dtype": np.float64,
}

ACTION_CFG: dict[str, Any] = {
    "lowercase": True,
    "strip_accents": "unicode",
    "analyzer": "word",
    "ngram_range": (1, 2),
    "token_pattern": r"(?u)\b\w\w+\b",
    "min_df": 2,
    "max_df": 1.0,
    "max_features": 8000,
    "sublinear_tf": True,
    "smooth_idf": True,
    "norm": "l2",
    "dtype": np.float64,
}

SOURCE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "minilm": (),
    "entities": (),
    "emotion": (),
    "word_1": ("word_1_base", "word_1_svd"),
    "word_1_2": ("word_1_2_base", "word_1_2_svd"),
    "word_1_3": ("word_1_3_base", "word_1_3_svd"),
    "char_3_5": ("char_3_5_base", "char_3_5_svd"),
    "topics": ("word_1_2_base", "topics_nmf"),
    "actions": ("actions_base", "actions_svd"),
    "linguistic": ("linguistic_scaler",),
}

SOURCE_ORDER = tuple(SOURCE_COMPONENTS)


def combine_blocks(block_dict: dict[str, np.ndarray], sources: Sequence[str]) -> np.ndarray:
    """Concatenate and L2-normalize representation blocks for a given feature bundle."""
    blocks = [safe_unit_rows(block_dict[s]) for s in sources]
    x = safe_unit_rows(np.concatenate(blocks, axis=1).astype(np.float32))
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError(f"Bloco combinado inválido em {sources}")
    return x


def bundle_component_time(
    pack: dict[str, Any],
    bundle: Sequence[str],
) -> tuple[float, float, list[str]]:
    """Aggregate deduplicated fit and transform seconds across feature components in a bundle."""
    names: list[str] = []
    for source in bundle:
        names.extend(SOURCE_COMPONENTS[source])
    unique = list(dict.fromkeys(names))
    fit = sum(float(pack["components"][c]["fit_seconds"]) for c in unique)
    transform = sum(float(pack["components"][c]["transform_seconds"]) for c in unique)
    return fit, transform, unique


def _fit_tfidf_base(
    texts_tr: list[str],
    texts_va: list[str],
    cfg: dict[str, Any],
) -> tuple[TfidfVectorizer, Any, Any, float, float]:
    vec = TfidfVectorizer(**cfg)
    t0 = time.perf_counter()
    a_tr = vec.fit_transform(texts_tr)
    fit_s = time.perf_counter() - t0
    if a_tr.shape[1] < 2:
        raise RuntimeError("Vocabulário TF-IDF insuficiente.")

    t1 = time.perf_counter()
    a_va = vec.transform(texts_va)
    transform_s = time.perf_counter() - t1
    return vec, a_tr, a_va, fit_s, transform_s


def _fit_svd(
    a_tr: Any,
    a_va: Any,
    target_dim: int,
    seed: int = CV_SEED,
) -> tuple[np.ndarray, np.ndarray, TruncatedSVD, float, float]:
    dim = min(target_dim, a_tr.shape[1] - 1, a_tr.shape[0] - 1)
    svd = TruncatedSVD(
        n_components=dim,
        algorithm="randomized",
        n_iter=7,
        n_oversamples=10,
        random_state=seed,
    )
    t0 = time.perf_counter()
    xtr = safe_unit_rows(svd.fit_transform(a_tr))
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    xva = safe_unit_rows(svd.transform(a_va))
    transform_s = time.perf_counter() - t1
    return xtr, xva, svd, fit_s, transform_s


def prepare_fixed_features(dataset: DatasetContext) -> dict[str, Any]:
    """Load or compute fold-independent frozen representations (MiniLM, Emotion, spaCy)."""
    dev = dataset.frame
    timing_rows = []

    t0 = time.perf_counter()
    x_minilm, minilm_meta = get_minilm_features(dev)
    timing_rows.append({
        "stage": "minilm",
        "observed_seconds": time.perf_counter() - t0,
        "cache_hit": minilm_meta.get("cache_hit"),
    })

    t0 = time.perf_counter()
    x_emotion, emotion_labels, emotion_meta = get_emotion_features(dev)
    timing_rows.append({
        "stage": "emotion",
        "observed_seconds": time.perf_counter() - t0,
        "cache_hit": emotion_meta.get("cache_hit"),
    })

    t0 = time.perf_counter()
    x_entities, entity_labels, action_docs, x_linguistic_raw, linguistic_names, spacy_meta = (
        get_spacy_text_views(dev)
    )
    timing_rows.append({
        "stage": "spacy_shared",
        "observed_seconds": time.perf_counter() - t0,
        "cache_hit": spacy_meta.get("cache_hit"),
    })

    n = len(dev)
    assert (
        len(x_minilm)
        == len(x_emotion)
        == len(x_entities)
        == len(action_docs)
        == len(x_linguistic_raw)
        == n
    )
    return {
        "arrays": {"minilm": x_minilm, "entities": x_entities, "emotion": x_emotion},
        "action_docs": action_docs,
        "linguistic_raw": x_linguistic_raw,
        "labels": {
            "emotion": emotion_labels,
            "entities": entity_labels,
            "linguistic": linguistic_names,
        },
        "meta": {"minilm": minilm_meta, "emotion": emotion_meta, "spacy": spacy_meta},
        "timing": pd.DataFrame(timing_rows),
    }


def build_fold_features(
    dataset: DatasetContext,
    fixed: dict[str, Any],
    tr: np.ndarray,
    va: np.ndarray,
    fold_number: int,
) -> dict[str, Any]:
    """Fit and transform fold-local lexical, topic, action, and linguistic representations."""
    texts_tr = dataset.frame.iloc[tr].texto.tolist()
    texts_va = dataset.frame.iloc[va].texto.tolist()
    arrays = fixed["arrays"]

    train = {
        "minilm": arrays["minilm"][tr],
        "entities": arrays["entities"][tr],
        "emotion": arrays["emotion"][tr],
    }
    valid = {
        "minilm": arrays["minilm"][va],
        "entities": arrays["entities"][va],
        "emotion": arrays["emotion"][va],
    }
    components: dict[str, dict[str, float]] = {}
    meta: dict[str, Any] = {"fold": fold_number}

    word_bases = {}
    for name, cfg in WORD_CFGS.items():
        _, a_tr, a_va, fit_b, trans_b = _fit_tfidf_base(texts_tr, texts_va, cfg)
        components[f"{name}_base"] = {"fit_seconds": fit_b, "transform_seconds": trans_b}
        xtr, xva, svd, fit_s, trans_s = _fit_svd(a_tr, a_va, LEX_SVD_DIM)
        components[f"{name}_svd"] = {"fit_seconds": fit_s, "transform_seconds": trans_s}
        train[name] = xtr
        valid[name] = xva
        meta[f"{name}_vocab"] = int(a_tr.shape[1])
        meta[f"{name}_dim"] = int(xtr.shape[1])
        meta[f"{name}_evr"] = float(svd.explained_variance_ratio_.sum())
        if name == "word_1_2":
            word_bases[name] = (a_tr, a_va)

    _, a_tr, a_va, fit_b, trans_b = _fit_tfidf_base(texts_tr, texts_va, CHAR_CFG)
    components["char_3_5_base"] = {"fit_seconds": fit_b, "transform_seconds": trans_b}
    xtr, xva, svd, fit_s, trans_s = _fit_svd(a_tr, a_va, LEX_SVD_DIM)
    components["char_3_5_svd"] = {"fit_seconds": fit_s, "transform_seconds": trans_s}
    train["char_3_5"] = xtr
    valid["char_3_5"] = xva
    meta["char_3_5_vocab"] = int(a_tr.shape[1])
    meta["char_3_5_dim"] = int(xtr.shape[1])
    meta["char_3_5_evr"] = float(svd.explained_variance_ratio_.sum())

    a_tr, a_va = word_bases["word_1_2"]
    nmf_dim = min(NMF_DIM, a_tr.shape[1], len(tr))
    nmf = NMF(
        n_components=nmf_dim,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=500,
        random_state=CV_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        xtr = safe_unit_rows(nmf.fit_transform(a_tr))
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        xva = safe_unit_rows(nmf.transform(a_va))
        trans_s = time.perf_counter() - t1
    components["topics_nmf"] = {"fit_seconds": fit_s, "transform_seconds": trans_s}
    train["topics"] = xtr
    valid["topics"] = xva
    meta["topics_dim"] = int(xtr.shape[1])

    actions_tr = [fixed["action_docs"][i] for i in tr]
    actions_va = [fixed["action_docs"][i] for i in va]
    _, a_tr, a_va, fit_b, trans_b = _fit_tfidf_base(actions_tr, actions_va, ACTION_CFG)
    components["actions_base"] = {"fit_seconds": fit_b, "transform_seconds": trans_b}
    xtr, xva, svd, fit_s, trans_s = _fit_svd(a_tr, a_va, ACTION_SVD_DIM)
    components["actions_svd"] = {"fit_seconds": fit_s, "transform_seconds": trans_s}
    train["actions"] = xtr
    valid["actions"] = xva
    meta["actions_vocab"] = int(a_tr.shape[1])
    meta["actions_dim"] = int(xtr.shape[1])
    meta["actions_evr"] = float(svd.explained_variance_ratio_.sum())

    scaler = StandardScaler()
    t0 = time.perf_counter()
    xtr = scaler.fit_transform(fixed["linguistic_raw"][tr]).astype(np.float32)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    xva = scaler.transform(fixed["linguistic_raw"][va]).astype(np.float32)
    trans_s = time.perf_counter() - t1
    components["linguistic_scaler"] = {"fit_seconds": fit_s, "transform_seconds": trans_s}
    train["linguistic"] = safe_unit_rows(xtr)
    valid["linguistic"] = safe_unit_rows(xva)
    meta["linguistic_dim"] = int(xtr.shape[1])

    for source in SOURCE_ORDER:
        if source not in train or train[source].shape[1] != valid[source].shape[1]:
            raise AssertionError(f"Fonte ausente ou dimensão instável: {source}")
        if not np.isfinite(train[source]).all() or not np.isfinite(valid[source]).all():
            raise ValueError(f"NaN/Inf em {source}")

    return {"train": train, "valid": valid, "components": components, "meta": meta}


def prepare_feature_context(dataset: DatasetContext) -> FeatureContext:
    """Orchestrate full feature extraction across frozen caches and fold-local splits."""
    print("Preparando extratores congelados/cacheados...")
    fixed = prepare_fixed_features(dataset)
    fixed["timing"].to_csv(STUDY_DIR / "fixed_extraction_timing.csv", index=False)

    fold_packs = []
    feature_meta = []
    component_rows = []
    print("Preparando transformações fold-local...")
    for fold, (tr, va) in enumerate(dataset.folds, start=1):
        pack = build_fold_features(dataset, fixed, tr, va, fold)
        fold_packs.append(pack)
        feature_meta.append(pack["meta"])
        for component, timing in pack["components"].items():
            component_rows.append({"fold": fold, "component": component, **timing})
        print(f"  fold {fold}/{len(dataset.folds)} pronto")

    feature_meta_df = pd.DataFrame(feature_meta)
    component_timing = pd.DataFrame(component_rows)
    feature_meta_df.to_csv(STUDY_DIR / "feature_meta_by_fold.csv", index=False)
    component_timing.to_csv(STUDY_DIR / "feature_component_timing.csv", index=False)

    catalog = []
    for source in SOURCE_ORDER:
        dims = sorted({p["train"][source].shape[1] for p in fold_packs})
        catalog.append({
            "source": source,
            "dim": dims[0] if len(dims) == 1 else str(dims),
            "fit_scope": "frozen/cache" if not SOURCE_COMPONENTS[source] else "outer-train",
            "components": "+".join(SOURCE_COMPONENTS[source]) or "precomputed",
        })
    catalog_df = pd.DataFrame(catalog)
    return FeatureContext(
        fold_packs=fold_packs,
        catalog=catalog_df,
        fixed_meta=fixed["meta"],
        fixed_extraction_timing=fixed["timing"],
        fold_meta=feature_meta_df,
        component_timing=component_timing,
    )
