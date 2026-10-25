"""Dataset loading, SHA-256 verification, and synopsis text cleaning."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from ..core.config import CLASSES, DATA, DATA_SHA, PARTS, SPLIT_SHA, TARGET


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


def load_development(
    data_path: str | Path = DATA,
    split_path: str | Path = PARTS,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load and validate the frozen development split and binary target matrix.

    Raises
    ------
    ValueError
        If the dataset or partition file SHA-256 does not match the frozen protocol hash.
    """
    if sha256(data_path) != DATA_SHA:
        raise ValueError("Dataset diferente da versão congelada.")
    if sha256(split_path) != SPLIT_SHA:
        raise ValueError("Partições diferentes da versão congelada.")

    raw = pd.read_csv(
        data_path,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    parts = pd.read_csv(split_path, dtype=str)

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

    assert len(dev) == 304 and dev.texto.str.len().gt(0).all()

    labels = [[x.strip() for x in s.split(",") if x.strip()] for s in dev[TARGET]]
    unknown = sorted(set(x for row in labels for x in row) - set(CLASSES))
    assert not unknown, f"Classes não previstas: {unknown}"

    mlb = MultiLabelBinarizer(classes=CLASSES)
    y = mlb.fit_transform(labels).astype(np.float32)
    return dev, y, parts
