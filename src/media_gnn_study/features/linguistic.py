"""SpaCy-based NER entity representations, narrative actions, and linguistic statistics."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from ..core.config import CACHE, DATA_SHA, ENTITY_HASH_DIM, SPACY_MODEL_ID
from ..core.graph import safe_unit_rows
from ..data.loader import clean_text


def _entity_key(text: str) -> str:
    cleaned = clean_text(text).lower().strip()
    cleaned = re.sub(r"[^\wÀ-ÿ]+", "_", cleaned, flags=re.UNICODE).strip("_")
    return cleaned or "vazio"


def _safe_lemma(token: Any) -> str:
    lemma = (token.lemma_ or token.text).lower().strip()
    return re.sub(r"[^\wÀ-ÿ]+", "_", lemma, flags=re.UNICODE).strip("_")


def get_spacy_text_views(
    dev: pd.DataFrame,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, list[str], dict[str, Any]]:
    """Run a single spaCy pass for NER, verb-object actions, and linguistic statistics.

    Reuses cached .npz files (including legacy caches) when available.
    """
    cache = CACHE / f"spacy_text_views_{SPACY_MODEL_ID}_v2.npz"
    meta_path = CACHE / f"spacy_text_views_{SPACY_MODEL_ID}_v2.json"

    if cache.is_file() and meta_path.is_file():
        with np.load(cache, allow_pickle=False) as z:
            ids = z["ids"].astype(str).tolist()
            x_ent = z["X_entities"].astype(np.float32)
            entity_labels = z["entity_labels"].astype(str).tolist()
            action_docs = z["action_docs"].astype(str).tolist()
            x_ling = z["X_linguistic_raw"].astype(np.float32)
            ling_names = z["linguistic_feature_names"].astype(str).tolist()
        if ids == dev.Id.tolist():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return x_ent, entity_labels, action_docs, x_ling, ling_names, {
                **meta,
                "cache_hit": True,
                "source": str(cache),
            }

    # Backward compatibility with v1 separate caches
    old_ent = CACHE / f"entidades_{SPACY_MODEL_ID}_{ENTITY_HASH_DIM}d_v1.npz"
    old_narr = CACHE / f"narrativa_{SPACY_MODEL_ID}_v1.npz"
    if old_ent.is_file() and old_narr.is_file():
        with np.load(old_ent, allow_pickle=False) as ze, np.load(old_narr, allow_pickle=False) as zn:
            ids_e = ze["ids"].astype(str).tolist()
            ids_n = zn["ids"].astype(str).tolist()
            if ids_e == dev.Id.tolist() == ids_n:
                x_ent = ze["X"].astype(np.float32)
                entity_labels = ze["labels"].astype(str).tolist()
                action_docs = zn["action_docs"].astype(str).tolist()
                x_ling = zn["X_linguistic_raw"].astype(np.float32)
                ling_names = zn["feature_names"].astype(str).tolist()

                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache,
                    ids=dev.Id.to_numpy(dtype=str),
                    X_entities=x_ent,
                    entity_labels=np.asarray(entity_labels, dtype=str),
                    action_docs=np.asarray(action_docs, dtype=str),
                    X_linguistic_raw=x_ling,
                    linguistic_feature_names=np.asarray(ling_names, dtype=str),
                )
                meta = {
                    "model": SPACY_MODEL_ID,
                    "data_sha256": DATA_SHA,
                    "rule": "reused_legacy_entity_and_narrative_caches",
                    "reused_legacy_caches": True,
                    "entity_dimensions": int(x_ent.shape[1]),
                    "linguistic_dimensions": int(x_ling.shape[1]),
                }
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                return x_ent, entity_labels, action_docs, x_ling, ling_names, {
                    **meta,
                    "cache_hit": True,
                    "source": "legacy_caches",
                }

    import spacy

    try:
        nlp = spacy.load(SPACY_MODEL_ID)
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL_ID], check=True)
        nlp = spacy.load(SPACY_MODEL_ID)

    entity_labels = sorted(nlp.get_pipe("ner").labels)
    label_to_i = {lab: i for i, lab in enumerate(entity_labels)}
    pos_names = ["VERB", "AUX", "NOUN", "PROPN", "ADJ", "ADV", "PRON"]
    ling_names = [
        "log_tokens",
        "log_sentences",
        "avg_sentence_len",
        "avg_alpha_token_len",
        "lexical_diversity",
        *[f"prop_{p.lower()}" for p in pos_names],
        "negation_ratio",
        "question_ratio",
        "exclamation_ratio",
        "punct_ratio",
        "verb_object_pairs_per_token",
    ]

    entity_docs, type_props, ent_stats, action_docs, ling_rows = [], [], [], [], []
    texts = dev.texto.tolist()
    for text, doc in zip(texts, nlp.pipe(texts, batch_size=32)):
        ents = list(doc.ents)
        counts = np.zeros(len(entity_labels), dtype=np.float32)
        ent_tokens, surfaces = [], []
        for ent in ents:
            key = _entity_key(ent.text)
            ent_tokens.append(f"{ent.label_}::{key}")
            surfaces.append(key)
            if ent.label_ in label_to_i:
                counts[label_to_i[ent.label_]] += 1

        total_ent = len(ents)
        n_word = max(1, len(re.findall(r"\w+", text, flags=re.UNICODE)))
        entity_docs.append(" ".join(ent_tokens))
        type_props.append(counts / max(total_ent, 1))
        ent_stats.append([
            np.log1p(total_ent),
            total_ent / n_word,
            len(set(surfaces)) / max(total_ent, 1),
            float(total_ent == 0),
        ])

        alpha = [t for t in doc if t.is_alpha]
        n_alpha = max(1, len(alpha))
        sentences = list(doc.sents)
        n_sent = max(1, len(sentences))
        pos_counts = {p: 0 for p in pos_names}
        lemmas = []
        for t in alpha:
            if t.pos_ in pos_counts:
                pos_counts[t.pos_] += 1
            lem = _safe_lemma(t)
            if lem:
                lemmas.append(lem)

        action_tokens = []
        pair_count = 0
        for t in alpha:
            if t.pos_ != "VERB":
                continue
            verb = _safe_lemma(t)
            if not verb:
                continue
            action_tokens.append(verb)
            for child in t.children:
                if child.dep_ in {"obj", "iobj", "obl"} and child.pos_ in {"NOUN", "PROPN", "PRON"}:
                    obj = _safe_lemma(child)
                    if obj:
                        action_tokens.append(f"{verb}__{obj}")
                        pair_count += 1

        action_docs.append(" ".join(action_tokens) if action_tokens else "semacao")
        token_count = max(1, len([t for t in doc if not t.is_space]))
        neg_count = sum(
            1 for t in doc if t.dep_ == "neg" or _safe_lemma(t) in {"nao", "não", "nunca", "jamais"}
        )
        punct_count = sum(1 for t in doc if t.is_punct)
        ling_rows.append([
            np.log1p(n_alpha),
            np.log1p(n_sent),
            n_alpha / n_sent,
            float(np.mean([len(t.text) for t in alpha])) if alpha else 0.0,
            len(set(lemmas)) / max(len(lemmas), 1),
            *[pos_counts[p] / n_alpha for p in pos_names],
            neg_count / token_count,
            text.count("?") / token_count,
            text.count("!") / token_count,
            punct_count / token_count,
            pair_count / n_alpha,
        ])

    hasher = HashingVectorizer(
        n_features=ENTITY_HASH_DIM,
        alternate_sign=False,
        norm=None,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
    )
    h = hasher.transform(entity_docs).toarray().astype(np.float32)
    x_ent = safe_unit_rows(
        np.concatenate(
            [h, np.asarray(type_props, np.float32), np.asarray(ent_stats, np.float32)],
            axis=1,
        )
    )
    x_ling = np.asarray(ling_rows, dtype=np.float32)
    if not np.isfinite(x_ling).all():
        raise ValueError("Features linguísticas inválidas.")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        ids=dev.Id.to_numpy(dtype=str),
        X_entities=x_ent,
        entity_labels=np.asarray(entity_labels, dtype=str),
        action_docs=np.asarray(action_docs, dtype=str),
        X_linguistic_raw=x_ling,
        linguistic_feature_names=np.asarray(ling_names, dtype=str),
    )
    meta = {
        "model": SPACY_MODEL_ID,
        "spacy_version": spacy.__version__,
        "data_sha256": DATA_SHA,
        "entity_rule": "entity_surface_hash_plus_type_proportions_and_counts_v1",
        "action_rule": "verb_lemmas_plus_simple_dependency_objects_v1",
        "linguistic_rule": "length_lexical_pos_negation_punctuation_v1",
        "entity_dimensions": int(x_ent.shape[1]),
        "linguistic_dimensions": int(x_ling.shape[1]),
        "reused_legacy_caches": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return x_ent, entity_labels, action_docs, x_ling, ling_names, {
        **meta,
        "cache_hit": False,
        "source": "computed",
    }
