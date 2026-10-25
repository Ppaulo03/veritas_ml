"""Pretrained neural text encoders (MiniLM and XLM-RoBERTa emotion) with disk caching.

Note: PyTorch and HuggingFace libraries are imported lazily only on cache misses.
When reading cached .npz embeddings, only NumPy is used.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import (
    CACHE,
    DEVICE_NAME,
    EMBED_RULE,
    EMOTION_MODEL_ID,
    MODEL_ID,
    MODEL_REVISION,
    REFERENCE_MINILM_RUN,
    ROOT,
)
from ..core.graph import safe_unit_rows


def encode_documents(texts: list[str], encoder: Any, batch_size: int = 32) -> np.ndarray:
    """Encode documents via non-overlapping token chunks weighted by content token count."""
    import torch
    from torch.nn import functional as F

    tok = encoder.tokenizer
    if not tok.is_fast:
        raise ValueError("É necessário tokenizer rápido.")

    limit = int(encoder.max_seq_length)
    counts = np.array([
        len(tok(t, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])
        for t in texts
    ], dtype=np.int64)
    if np.any(counts == 0):
        raise ValueError("Sinopse sem tokens.")

    packed = tok(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=limit,
        stride=0,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True,
        return_attention_mask=True,
        padding=True,
        return_tensors="pt",
    )
    owner = np.asarray(packed.pop("overflow_to_sample_mapping"), dtype=np.int64)
    special_mask = packed.pop("special_tokens_mask")
    weights = (packed["attention_mask"] * (1 - special_mask)).sum(dim=1).numpy().astype(np.int64)
    covered = np.bincount(owner, weights=weights, minlength=len(texts)).astype(np.int64)
    assert np.array_equal(covered, counts), "Token perdido ou duplicado."

    encoder.eval()
    encoder.requires_grad_(False)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(owner), batch_size):
            features = {
                k: v[start : start + batch_size].to(encoder.device)
                for k, v in packed.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }
            z = encoder(features)["sentence_embedding"]
            chunks.append(F.normalize(z.float(), p=2, dim=1).cpu().numpy())

    chunks_arr = np.concatenate(chunks)
    x = np.zeros((len(texts), chunks_arr.shape[1]), dtype=np.float32)
    np.add.at(x, owner, chunks_arr * weights[:, None])
    x /= counts[:, None]
    return safe_unit_rows(x)


def get_minilm_features(dev: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Retrieve or compute frozen MiniLM embeddings for development synopses."""
    cache_path = CACHE / "minilm_embeddings.npz"
    legacy = ROOT / "03_resultados" / "02_mvp_mlp_vs_gnn" / REFERENCE_MINILM_RUN / "embeddings_dev.npz"

    for source in (cache_path, legacy):
        if source.is_file():
            with np.load(source, allow_pickle=False) as z:
                ids = z["ids"].astype(str).tolist()
                x = z["X"].astype(np.float32)
            if ids == dev.Id.tolist() and x.shape == (len(dev), 384):
                x = safe_unit_rows(x)
                if source != cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache_path, ids=dev.Id.to_numpy(dtype=str), X=x)
                return x, {
                    "cache_hit": True,
                    "source": str(source),
                    "model": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "rule": EMBED_RULE,
                }

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        trust_remote_code=False,
    )
    x = encode_documents(dev.texto.tolist(), encoder)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=dev.Id.to_numpy(dtype=str), X=x)
    return x, {
        "cache_hit": False,
        "source": "computed",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "rule": EMBED_RULE,
        "device": "cpu",
    }


def _resolve_hf_revision(model_id: str) -> str:
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha
    except Exception:
        return "main"


def _encode_emotion_chunks(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    batch_size: int = 16,
) -> np.ndarray:
    import torch

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Tokenizer rápido necessário para agregação de chunks de emoção.")

    max_len = int(getattr(tokenizer, "model_max_length", 512))
    if max_len > 4096 or max_len < 16:
        max_len = 512

    packed = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        stride=0,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True,
        return_attention_mask=True,
        padding=True,
        return_tensors="pt",
    )
    owner = np.asarray(packed.pop("overflow_to_sample_mapping"), dtype=np.int64)
    special = packed.pop("special_tokens_mask")
    weights = (packed["attention_mask"] * (1 - special)).sum(dim=1).numpy().astype(np.float32)

    probs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(owner), batch_size):
            batch = {
                k: v[start : start + batch_size].to(DEVICE_NAME)
                for k, v in packed.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }
            probs.append(torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy())

    probs_arr = np.concatenate(probs).astype(np.float32)
    x = np.zeros((len(texts), probs_arr.shape[1]), dtype=np.float32)
    den = np.zeros(len(texts), dtype=np.float32)
    np.add.at(x, owner, probs_arr * weights[:, None])
    np.add.at(den, owner, weights)
    x /= np.maximum(den[:, None], 1e-8)
    return x


def get_emotion_features(dev: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Retrieve or compute frozen XLM-RoBERTa emotion probabilities."""
    cache = CACHE / "emocao_xlm_emo_t_v1.npz"
    meta_path = CACHE / "emocao_xlm_emo_t_v1.json"

    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            ids = z["ids"].astype(str).tolist()
            x = z["X"].astype(np.float32)
            labels = z["labels"].astype(str).tolist()
        if ids == dev.Id.tolist() and len(x) == len(dev):
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            return x, labels, {**meta, "cache_hit": True, "source": str(cache)}

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    revision = _resolve_hf_revision(EMOTION_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        EMOTION_MODEL_ID,
        revision=revision,
        use_fast=True,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL_ID,
        revision=revision,
        trust_remote_code=False,
    ).to(DEVICE_NAME)

    x = _encode_emotion_chunks(dev.texto.tolist(), tokenizer, model)
    labels = [str(model.config.id2label[i]) for i in range(model.config.num_labels)]

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        ids=dev.Id.to_numpy(dtype=str),
        X=x,
        labels=np.asarray(labels, dtype=str),
    )
    meta = {
        "model": EMOTION_MODEL_ID,
        "resolved_revision": revision,
        "labels": labels,
        "dimensions": int(x.shape[1]),
        "aggregation": "nonoverlapping_token_chunks_content_weighted_mean_softmax_v1",
        "domain_note": "multilingual social-media emotion model used frozen on synopses",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return x, labels, {**meta, "cache_hit": False, "source": "computed", "device": "cpu"}
