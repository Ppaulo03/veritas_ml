"""Dataset loading, dynamic SHA-256 verification, and text cleaning."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from ..core.config import CV_SEED, DATA, PARTS


def sha256(path: str | Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_text(text: str) -> str:
    """Strip standard editorial footers (e.g., MAL Rewrite) and normalize whitespace."""
    cleaned = re.sub(
        r"\[\s*(?:Escrito por MAL Reescrever|Written by MAL Rewrite)\s*\]",
        " ",
        str(text),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _infer_franchise_group(title: str) -> str:
    """Derive a deterministic canonical group key from a title for local fallback splits."""
    norm = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode("ascii").lower()
    head = re.split(r"[:\-\(\[]|\b(?:parte|part|season|temporada|vol|filme|movie)\b", norm)[0]
    tokens = re.findall(r"[a-z0-9]+", head)
    if not tokens:
        tokens = re.findall(r"[a-z0-9]+", norm)
    return "_".join(tokens[:3]) if tokens else "grupo_unico"


def build_local_fallback_partitions(
    raw: pd.DataFrame,
    target_dev_fraction: float = 0.75,
    seed: int = CV_SEED,
) -> pd.DataFrame:
    """Create a deterministic group-disjoint development/audit partition when no split file is provided."""
    name_col = "Nome" if "Nome" in raw.columns else raw.columns[0]
    df = raw[["Id"]].copy()
    df["grupo_franquia"] = (
        raw["grupo_franquia"].astype(str)
        if "grupo_franquia" in raw.columns
        else raw[name_col].map(_infer_franchise_group)
    )

    target_dev = max(1, int(round(len(raw) * target_dev_fraction)))
    group_sizes = df.groupby("grupo_franquia").size().rename("size").reset_index()
    rng = np.random.default_rng(seed)
    group_sizes["tie"] = rng.random(len(group_sizes))
    group_sizes = group_sizes.sort_values(["size", "tie"], ascending=[False, True]).reset_index(drop=True)

    dev_groups: set[str] = set()
    dev_count = 0
    for row in group_sizes.itertuples(index=False):
        if dev_count + int(row.size) <= target_dev or not dev_groups:
            dev_groups.add(str(row.grupo_franquia))
            dev_count += int(row.size)

    df["particao"] = np.where(df["grupo_franquia"].isin(dev_groups), "desenvolvimento", "auditoria")
    return df[["Id", "particao", "grupo_franquia"]]


def load_development(
    data_path: str | Path = DATA,
    split_path: str | Path | None = PARTS,
    *,
    target_col: str | None = None,
    text_col: str | None = None,
    classes: Sequence[str] | None = None,
    expected_data_sha: str | None = None,
    expected_split_sha: str | None = None,
    allow_local_fallback: bool = False,
    sep: str = ";",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load and validate a development split and binary multilabel target matrix.

    Computes SHA-256 digests dynamically and optionally verifies them against
    ``expected_data_sha`` and ``expected_split_sha`` when provided.
    """
    resolved_data = Path(data_path)
    if not resolved_data.is_file():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em '{resolved_data}'.")

    actual_data_sha = sha256(resolved_data)
    if expected_data_sha is not None and actual_data_sha != expected_data_sha:
        raise ValueError("Dataset diferente do hash SHA-256 esperado.")

    raw = pd.read_csv(
        resolved_data,
        sep=sep,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    resolved_split = Path(split_path) if split_path is not None else None
    if resolved_split is not None and resolved_split.is_file():
        actual_split_sha = sha256(resolved_split)
        if expected_split_sha is not None and actual_split_sha != expected_split_sha:
            raise ValueError("Partições diferentes do hash SHA-256 esperado.")
        parts = pd.read_csv(resolved_split, dtype=str)
    elif allow_local_fallback:
        parts = build_local_fallback_partitions(raw)
    else:
        raise FileNotFoundError(
            f"Arquivo de partições não encontrado em '{resolved_split}'. "
            "Passe allow_local_fallback=True para gerar partição determinística local."
        )

    assert raw.Id.is_unique and parts.Id.is_unique and set(raw.Id) == set(parts.Id)
    assert set(parts.particao).issubset({"desenvolvimento", "auditoria"})

    g_dev = set(parts.loc[parts.particao == "desenvolvimento", "grupo_franquia"])
    g_aud = set(parts.loc[parts.particao == "auditoria", "grupo_franquia"])
    assert not (g_dev & g_aud), "Group leakage detected between development and audit sets."

    resolved_target = target_col or (
        "Macrogênero (Gênero unificado)"
        if "Macrogênero (Gênero unificado)" in raw.columns
        else "target"
    )
    resolved_text = text_col or ("Sinopse" if "Sinopse" in raw.columns else "texto")

    dev_ids = set(parts.loc[parts.particao == "desenvolvimento", "Id"])
    keep_cols = [c for c in ("Id", "Nome", resolved_text, "Origem", resolved_target) if c in raw.columns]
    dev = raw.loc[raw.Id.isin(dev_ids), keep_cols].copy()
    dev = dev.merge(parts[["Id", "grupo_franquia"]], on="Id", validate="one_to_one")
    dev = dev.sort_values("Id").reset_index(drop=True)
    dev["texto"] = dev[resolved_text].map(clean_text)

    assert len(dev) > 0 and dev.texto.str.len().gt(0).all()

    labels = [[x.strip() for x in s.split(",") if x.strip()] for s in dev[resolved_target]]
    resolved_classes = (
        list(classes)
        if classes is not None
        else sorted({x for row in labels for x in row})
    )
    unknown = sorted({x for row in labels for x in row} - set(resolved_classes))
    assert not unknown, f"Classes não previstas: {unknown}"

    mlb = MultiLabelBinarizer(classes=resolved_classes)
    y = mlb.fit_transform(labels).astype(np.float32)
    return dev, y, parts
