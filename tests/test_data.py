"""Tests for data loading utilities and franchise-grouped fold splits."""

from pathlib import Path
from media_gnn_study.core.config import DATA_SHA
from media_gnn_study.data import (
    build_dataset_context,
    clean_text,
    make_grouped_folds,
    sha256,
)


def test_clean_text_strips_mal_rewrite_and_normalizes_spaces():
    raw = "Sinopse legal   com espaços.\n\n[Escrito por MAL Reescrever]"
    assert clean_text(raw) == "Sinopse legal com espaços."


def test_local_dataset_matches_frozen_sha256_and_builds_local_context():
    csv_path = Path("medias_classificadas.csv")
    if csv_path.is_file():
        assert sha256(csv_path) == DATA_SHA
        dataset = build_dataset_context(
            data_path=csv_path,
            persist_folds=False,
            allow_local_fallback=True,
        )
        assert len(dataset.frame) == 304
        assert dataset.targets.shape == (304, len(dataset.classes))
        assert len(dataset.folds) == 5


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
