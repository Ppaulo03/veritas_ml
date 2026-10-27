"""Tests for data loading utilities, dynamic SHA-256 verification, and grouped fold splits."""

from pathlib import Path
import pytest
from veritas_ml.data import (
    build_dataset_context,
    clean_text,
    make_grouped_folds,
    sha256,
)


def test_clean_text_strips_mal_rewrite_and_normalizes_spaces():
    raw = "Sinopse legal   com espaços.\n\n[Escrito por MAL Reescrever]"
    assert clean_text(raw) == "Sinopse legal com espaços."


def test_dynamic_sha256_and_dataset_context_from_custom_csv(tmp_path: Path):
    csv_path = tmp_path / "custom_data.csv"
    rows = [
        "Id;Nome;texto;target;grupo_franquia",
        "1;Item 1;Texto sobre espaço e galáxia;SciFi, Ação;g1",
        "2;Item 2;Texto sobre planeta e nave;SciFi;g2",
        "3;Item 3;Texto sobre romance e escola;Romance;g3",
        "4;Item 4;Texto sobre amizade e escola;Romance, Drama;g4",
        "5;Item 5;Texto sobre investigação e crime;Mistério, Drama;g5",
        "6;Item 6;Texto sobre pista e detetive;Mistério;g6",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    digest = sha256(csv_path)

    dataset = build_dataset_context(
        data_path=csv_path,
        persist_folds=False,
        allow_local_fallback=True,
        expected_data_sha=digest,
        n_splits=2,
    )
    assert dataset.metadata["data_sha256"] == digest
    assert dataset.classes == ("Ação", "Drama", "Mistério", "Romance", "SciFi")
    assert len(dataset.folds) == 2

    with pytest.raises(ValueError, match="SHA-256"):
        build_dataset_context(
            data_path=csv_path,
            persist_folds=False,
            allow_local_fallback=True,
            expected_data_sha="0000_invalid_hash",
            n_splits=2,
        )


def test_make_grouped_folds_prevents_franchise_leakage():
    groups = (
        ["franquia_A"] * 5
        + ["franquia_B"] * 4
        + ["franquia_C"] * 3
        + ["franquia_D"] * 2
        + ["franquia_E"] * 2
    )
    folds, fold_id = make_grouped_folds(groups, n_splits=3, seed=42)
    assert len(folds) == 3
    assert len(fold_id) == len(groups)

    for tr, va in folds:
        tr_groups = {groups[i] for i in tr}
        va_groups = {groups[i] for i in va}
        assert tr_groups.isdisjoint(va_groups)
