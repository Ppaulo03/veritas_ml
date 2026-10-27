"""Group-disjoint cross-validation folds and DatasetContext assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.config import (
    CV_SEED,
    CV_SPLITS,
    DATA,
    PARTS,
    STUDY_DIR,
    ensure_runtime_dirs,
)
from ..core.contracts import DatasetContext
from .loader import _parse_multilabel_cell, _truthy_mask, load_development, sha256


def make_grouped_folds(
    groups: Sequence[str] | pd.Series,
    n_splits: int = CV_SPLITS,
    seed: int = CV_SEED,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Build greedy size-balanced group folds ensuring zero group overlap across splits."""
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
    *,
    target_col: str | None = None,
    text_col: str | None = None,
    classes: Sequence[str] | None = None,
    expected_data_sha: str | None = None,
    expected_split_sha: str | None = None,
    exclude_col: str | None = None,
    allow_local_fallback: bool = False,
    n_splits: int = CV_SPLITS,
    seed: int = CV_SEED,
    sep: str = ";",
) -> DatasetContext:
    """Assemble DatasetContext from CSV files, computing SHA-256 digests dynamically."""
    resolved_data = Path(data_path or DATA)
    resolved_split = Path(split_path or PARTS)

    dev, y, partitions = load_development(
        resolved_data,
        resolved_split,
        target_col=target_col,
        text_col=text_col,
        classes=classes,
        expected_data_sha=expected_data_sha,
        expected_split_sha=expected_split_sha,
        exclude_col=exclude_col,
        allow_local_fallback=allow_local_fallback,
        sep=sep,
    )
    folds, fold_id = make_grouped_folds(dev.grupo_franquia, n_splits=n_splits, seed=seed)

    actual_data_sha = sha256(resolved_data)
    actual_split_sha = (
        sha256(resolved_split) if resolved_split.is_file() else "LOCAL_FALLBACK_SPLIT"
    )
    resolved_target = target_col or (
        "Macrogênero (Gênero unificado)"
        if "Macrogênero (Gênero unificado)" in dev.columns
        else "target"
    )
    if classes is not None:
        resolved_classes = tuple(classes)
    else:
        labels = [_parse_multilabel_cell(s) for s in dev[resolved_target]]
        resolved_classes = tuple(sorted({x for row in labels for x in row}))

    if exclude_col is not None:
        excluded_mask = partitions["particao"].eq("desenvolvimento") & _truthy_mask(
            partitions[exclude_col]
        )
        n_excluded = int(excluded_mask.sum())
    else:
        n_excluded = 0

    dataset = DatasetContext(
        frame=dev,
        targets=y,
        classes=resolved_classes,
        folds=tuple(folds),
        fold_id=fold_id,
        partitions=partitions,
        metadata={
            "data_sha256": actual_data_sha,
            "split_sha256": actual_split_sha,
            "n_development": int(len(dev)),
            "n_audit_reserved": int((partitions.particao == "auditoria").sum()),
            "n_excluded": n_excluded,
            "exclude_column": exclude_col,
            "target_column": resolved_target,
            "group_column": "grupo_franquia",
        },
    )
    if persist_folds:
        ensure_runtime_dirs()
        fold_df = pd.DataFrame({"Id": dataset.frame.Id, "fold": dataset.fold_id + 1})
        fold_df.to_csv(STUDY_DIR / "folds.csv", index=False)

    return dataset


def fold_summary(dataset: DatasetContext) -> pd.DataFrame:
    """Generate a summary table of sample counts, unique groups, and minimum class support per fold."""
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
