"""Franchise-grouped cross-validation folds and DatasetContext assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.config import (
    CLASSES,
    CV_SEED,
    CV_SPLITS,
    DATA,
    DATA_SHA,
    PARTS,
    SPLIT_SHA,
    STUDY_DIR,
    TARGET,
    ensure_runtime_dirs,
)
from ..core.contracts import DatasetContext
from .loader import load_development


def make_grouped_folds(
    groups: Sequence[str] | pd.Series,
    n_splits: int = CV_SPLITS,
    seed: int = CV_SEED,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Build greedy size-balanced group folds ensuring zero franchise overlap across splits."""
    groups_series = pd.Series(groups, dtype=str).reset_index(drop=True)
    sizes = groups_series.value_counts().rename_axis("group").reset_index(name="size")

    rng = np.random.default_rng(seed)
    sizes["tie"] = rng.random(len(sizes))
    sizes = sizes.sort_values(["size", "tie"], ascending=[False, True]).reset_index(drop=True)

    fold_sizes = np.zeros(n_splits, dtype=int)
    assignment: dict[str, int] = {}
    for row in sizes.itertuples(index=False):
        target_fold = int(np.argmin(fold_sizes))
        assignment[row.group] = target_fold
        fold_sizes[target_fold] += int(row.size)

    fold_id = groups_series.map(assignment).to_numpy()
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_splits):
        va = np.where(fold_id == f)[0]
        tr = np.where(fold_id != f)[0]
        assert not (set(groups_series.iloc[tr]) & set(groups_series.iloc[va]))
        folds.append((tr, va))

    return folds, fold_id


def build_dataset_context(
    data_path: str | Path | None = None,
    split_path: str | Path | None = None,
    persist_folds: bool = True,
) -> DatasetContext:
    """Assemble DatasetContext from frozen files and optionally persist fold assignments."""
    ensure_runtime_dirs()
    resolved_data = Path(data_path or DATA)
    resolved_split = Path(split_path or PARTS)

    dev, y, partitions = load_development(resolved_data, resolved_split)
    folds, fold_id = make_grouped_folds(dev.grupo_franquia)

    dataset = DatasetContext(
        frame=dev,
        targets=y,
        classes=tuple(CLASSES),
        folds=tuple(folds),
        fold_id=fold_id,
        partitions=partitions,
        metadata={
            "data_sha256": DATA_SHA,
            "split_sha256": SPLIT_SHA,
            "n_development": int(len(dev)),
            "n_audit_reserved": int((partitions.particao == "auditoria").sum()),
            "target_column": TARGET,
            "group_column": "grupo_franquia",
        },
    )
    if persist_folds:
        fold_df = pd.DataFrame({"Id": dataset.frame.Id, "fold": dataset.fold_id + 1})
        fold_df.to_csv(STUDY_DIR / "folds.csv", index=False)

    return dataset


def fold_summary(dataset: DatasetContext) -> pd.DataFrame:
    """Generate a summary table of sample counts, unique franchises, and minimum class support per fold."""
    rows = []
    for fold, (tr, va) in enumerate(dataset.folds, start=1):
        support = dataset.targets[tr].sum(axis=0)
        rows.append({
            "fold": fold,
            "n_treino": len(tr),
            "n_validacao": len(va),
            "grupos_validacao": dataset.frame.iloc[va].grupo_franquia.nunique(),
            "menor_suporte_treino": int(support.min()),
        })
    return pd.DataFrame(rows)
