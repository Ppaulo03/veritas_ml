"""Dataset loading, SHA-256 verification, and synopsis text cleaning."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from ..core.config import CLASSES, CV_SEED, DATA, DATA_SHA, PARTS, ROOT, SPLIT_SHA, TARGET


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
    """Derive a deterministic canonical franchise key from a media title for local fallback splits."""
    norm = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode("ascii").lower()
    head = re.split(r"[:\-\(\[]|\b(?:parte|part|season|temporada|vol|filme|movie)\b", norm)[0]
    tokens = re.findall(r"[a-z0-9]+", head)
    if not tokens:
        tokens = re.findall(r"[a-z0-9]+", norm)
    return "_".join(tokens[:3]) if tokens else "obra_unica"


def build_local_fallback_partitions(raw: pd.DataFrame, target_dev: int = 304, seed: int = CV_SEED) -> pd.DataFrame:
    """Create a deterministic franchise-disjoint development/audit partition when particoes_franquias.csv is absent."""
    df = raw[["Id", "Nome"]].copy()
    df["grupo_franquia"] = df["Nome"].map(_infer_franchise_group)

    group_sizes = df.groupby("grupo_franquia").size().rename("size").reset_index()
    rng = np.random.default_rng(seed)
    group_sizes["tie"] = rng.random(len(group_sizes))
    group_sizes = group_sizes.sort_values(["size", "tie"], ascending=[False, True]).reset_index(drop=True)

    dev_groups: set[str] = set()
    dev_count = 0
    for row in group_sizes.itertuples(index=False):
        if dev_count + int(row.size) <= target_dev:
            dev_groups.add(str(row.grupo_franquia))
            dev_count += int(row.size)

    df["particao"] = np.where(df["grupo_franquia"].isin(dev_groups), "desenvolvimento", "auditoria")
    return df[["Id", "particao", "grupo_franquia"]]


def _resolve_data_path(data_path: str | Path) -> Path:
    p = Path(data_path)
    if p.is_file():
        return p
    fallback = ROOT / "medias_classificadas.csv"
    if fallback.is_file():
        return fallback
    return p


def load_development(
    data_path: str | Path = DATA,
    split_path: str | Path = PARTS,
    *,
    allow_local_fallback: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load and validate the frozen development split and binary target matrix.

    Raises
    ------
    ValueError
        If the dataset or partition file SHA-256 does not match the frozen protocol hash.
    """
    resolved_data = _resolve_data_path(data_path)
    if sha256(resolved_data) != DATA_SHA:
        raise ValueError("Dataset diferente da versão congelada.")

    raw = pd.read_csv(
        resolved_data,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    resolved_split = Path(split_path)
    if resolved_split.is_file():
        if sha256(resolved_split) != SPLIT_SHA:
            raise ValueError("Partições diferentes da versão congelada.")
        parts = pd.read_csv(resolved_split, dtype=str)
    elif allow_local_fallback:
        parts = build_local_fallback_partitions(raw)
    else:
        raise FileNotFoundError(
            f"Arquivo de partições não encontrado em '{resolved_split}'. "
            "Passe allow_local_fallback=True para gerar partição determinística local."
        )

    assert raw.Id.is_unique and parts.Id.is_unique and set(raw.Id) == set(parts.Id)
    assert set(parts.particao) == {"desenvolvimento", "auditoria"}

    g_dev = set(parts.loc[parts.particao == "desenvolvimento", "grupo_franquia"])
    g_aud = set(parts.loc[parts.particao == "auditoria", "grupo_franquia"])
    assert not (g_dev & g_aud), "Franchise leakage detected between development and audit sets."

    dev_ids = set(parts.loc[parts.particao == "desenvolvimento", "Id"])
    dev = raw.loc[raw.Id.isin(dev_ids), ["Id", "Nome", "Sinopse", "Origem", TARGET]].copy()
    dev = dev.merge(parts[["Id", "grupo_franquia"]], on="Id", validate="one_to_one")
    dev = dev.sort_values("Id").reset_index(drop=True)
    dev["texto"] = dev.Sinopse.map(clean_text)

    if not allow_local_fallback:
        assert len(dev) == 304
    assert dev.texto.str.len().gt(0).all()

    labels = [[x.strip() for x in s.split(",") if x.strip()] for s in dev[TARGET]]
    unknown = sorted(set(x for row in labels for x in row) - set(CLASSES))
    assert not unknown, f"Classes não previstas: {unknown}"

    mlb = MultiLabelBinarizer(classes=CLASSES)
    y = mlb.fit_transform(labels).astype(np.float32)
    return dev, y, parts
