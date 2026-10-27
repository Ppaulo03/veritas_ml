"""Tests for data loading utilities, dynamic SHA-256 verification, exclusion masks, and grouped fold splits."""

from pathlib import Path
import numpy as np
import pytest
from veritas_ml.data import (
    build_dataset_context,
    clean_text,
    make_grouped_folds,
    sha256,
)
from veritas_ml.data.loader import _parse_multilabel_cell


def test_clean_text_strips_mal_rewrite_and_normalizes_spaces():
    raw = "Sinopse legal   com espaços.\n\n[Escrito por MAL Reescrever]"
    assert clean_text(raw) == "Sinopse legal com espaços."


def test_parse_multilabel_cell_handles_braces_spaces_single_and_empty():
    assert _parse_multilabel_cell("Ação,Drama") == ["Ação", "Drama"]
    assert _parse_multilabel_cell("{Ação,Drama}") == ["Ação", "Drama"]
    assert _parse_multilabel_cell("  { Ação ,  Drama }  ") == ["Ação", "Drama"]
    assert _parse_multilabel_cell("Ação") == ["Ação"]
    assert _parse_multilabel_cell("{Ação}") == ["Ação"]
    assert _parse_multilabel_cell("") == []
    assert _parse_multilabel_cell("   ") == []
    assert _parse_multilabel_cell("{}") == []
    assert _parse_multilabel_cell("{ ,  }") == []


def test_dataset_context_accepts_braced_multilabel_and_rejects_unknown_classes(tmp_path: Path):
    csv_path = tmp_path / "braced_data.csv"
    parts_path = tmp_path / "braced_parts.csv"
    rows = [
        "Id;Nome;Sinopse;Macrogênero (Gênero unificado)",
        "1;Obra 1;Sinopse um;{Ação,Drama}",
        "2;Obra 2;Sinopse dois;{ Ação, SciFi }",
        "3;Obra 3;Sinopse três;{Drama}",
        "4;Obra 4;Sinopse quatro;SciFi, Drama",
    ]
    parts_rows = [
        "Id,particao,grupo_franquia",
        "1,desenvolvimento,g1",
        "2,desenvolvimento,g2",
        "3,desenvolvimento,g3",
        "4,desenvolvimento,g4",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    parts_path.write_text("\n".join(parts_rows), encoding="utf-8")

    dataset = build_dataset_context(
        data_path=csv_path,
        split_path=parts_path,
        persist_folds=False,
        classes=("Ação", "Drama", "SciFi"),
        n_splits=2,
    )
    assert dataset.classes == ("Ação", "Drama", "SciFi")
    expected_y = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(dataset.targets, expected_y)

    # Auto-inferred classes when classes=None must also match without braces
    dataset_auto = build_dataset_context(
        data_path=csv_path,
        split_path=parts_path,
        persist_folds=False,
        n_splits=2,
    )
    assert dataset_auto.classes == ("Ação", "Drama", "SciFi")

    # Unknown class must still be strictly rejected
    with pytest.raises(AssertionError, match="Classes não previstas"):
        build_dataset_context(
            data_path=csv_path,
            split_path=parts_path,
            persist_folds=False,
            classes=("Ação", "Drama"),
            n_splits=2,
        )


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
    assert dataset.metadata["n_excluded"] == 0
    assert dataset.metadata["exclude_column"] is None
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


def test_exclude_col_filters_blocked_development_items_and_prevents_group_leakage(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    parts_path = tmp_path / "parts.csv"

    csv_rows = [
        "Id;Nome;Sinopse;Macrogênero (Gênero unificado)",
        "1;Obra 1;Sinopse um;Ação, SciFi",
        "2;Obra 2;Sinopse dois;SciFi",
        "3;Obra 3;Sinopse três;Drama",
        "4;Obra 4;Sinopse quatro;Drama, Ação",
        "5;Obra 5 (Franquia Audit);Sinopse cinco;Ação",
        "6;Obra 6 (Audit);Sinopse seis;Drama",
    ]
    parts_rows = [
        "Id,particao,grupo_franquia,bloqueado_audit_franquia",
        "1,desenvolvimento,g1,False",
        "2,desenvolvimento,g2,False",
        "3,desenvolvimento,g3,False",
        "4,desenvolvimento,g4,False",
        "5,desenvolvimento,g_audit,True",
        "6,auditoria,g_audit,False",
    ]
    csv_path.write_text("\n".join(csv_rows), encoding="utf-8")
    parts_path.write_text("\n".join(parts_rows), encoding="utf-8")

    # Without exclude_col, item 5 in 'desenvolvimento' shares 'g_audit' with item 6 in 'auditoria'
    with pytest.raises(AssertionError, match="Group leakage"):
        build_dataset_context(
            data_path=csv_path,
            split_path=parts_path,
            persist_folds=False,
            n_splits=2,
        )

    # With exclude_col="bloqueado_audit_franquia", item 5 is excluded from development
    dataset = build_dataset_context(
        data_path=csv_path,
        split_path=parts_path,
        persist_folds=False,
        exclude_col="bloqueado_audit_franquia",
        expected_data_sha=sha256(csv_path),
        expected_split_sha=sha256(parts_path),
        n_splits=2,
    )
    assert len(dataset.frame) == 4
    assert set(dataset.frame["Id"]) == {"1", "2", "3", "4"}
    assert dataset.metadata["n_development"] == 4
    assert dataset.metadata["n_audit_reserved"] == 1
    assert dataset.metadata["n_excluded"] == 1
    assert dataset.metadata["exclude_column"] == "bloqueado_audit_franquia"

    with pytest.raises(ValueError, match="Coluna de exclusão ausente"):
        build_dataset_context(
            data_path=csv_path,
            split_path=parts_path,
            persist_folds=False,
            exclude_col="coluna_inexistente",
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
