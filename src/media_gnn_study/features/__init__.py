"""Feature extraction, caching, and fold-local representation assembly."""

from ..core.graph import safe_unit_rows
from .encoders import (
    _encode_emotion_chunks,
    encode_documents,
    get_emotion_features,
    get_minilm_features,
)
from .lexical import (
    ACTION_CFG,
    BASE_TFIDF_CFG,
    CHAR_CFG,
    SOURCE_COMPONENTS,
    SOURCE_ORDER,
    WORD_CFGS,
    _fit_svd,
    _fit_tfidf_base,
    build_fold_features,
    bundle_component_time,
    combine_blocks,
    prepare_feature_context,
    prepare_fixed_features,
)
from .linguistic import get_spacy_text_views

__all__ = [
    "safe_unit_rows",
    "encode_documents",
    "get_minilm_features",
    "_encode_emotion_chunks",
    "get_emotion_features",
    "get_spacy_text_views",
    "BASE_TFIDF_CFG",
    "WORD_CFGS",
    "CHAR_CFG",
    "ACTION_CFG",
    "SOURCE_COMPONENTS",
    "SOURCE_ORDER",
    "_fit_tfidf_base",
    "_fit_svd",
    "combine_blocks",
    "bundle_component_time",
    "prepare_fixed_features",
    "build_fold_features",
    "prepare_feature_context",
]
