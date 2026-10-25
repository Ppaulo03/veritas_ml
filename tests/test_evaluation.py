"""Tests for evaluation diagnostics, StudyResult assembly, and renderer validation."""

import numpy as np
import pandas as pd
from veritas_ml.core.contracts import DatasetContext
from veritas_ml.engine import build_experiment_plan, standardize_experiment_result
from veritas_ml.evaluation import (
    assemble_study_result,
    build_dataset_profile,
    pareto_front,
    validate_study_result,
)


def _make_dummy_dataset() -> DatasetContext:
    frame = pd.DataFrame({
        "Id": ["w1", "w2", "w3", "w4"],
        "Nome": ["Obra 1", "Obra 2", "Obra 3", "Obra 4"],
        "Sinopse": ["a", "b", "c", "d"],
        "Origem": ["human"] * 4,
        "Macrogênero (Gênero unificado)": ["Ação", "Comédia", "Ação, Comédia", "Drama"],
        "grupo_franquia": ["g1", "g2", "g1", "g2"],
        "texto": ["a", "b", "c", "d"],
    })
    targets = np.array(
        [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    folds = ((np.array([0, 2]), np.array([1, 3])), (np.array([1, 3]), np.array([0, 2])))
    fold_id = np.array([0, 1, 0, 1])
    partitions = pd.DataFrame({"Id": ["w1", "w2", "w3", "w4"], "particao": ["desenvolvimento"] * 4})
    return DatasetContext(
        frame=frame,
        targets=targets,
        classes=("Ação", "Comédia", "Drama"),
        folds=folds,
        fold_id=fold_id,
        partitions=partitions,
        metadata={"n_development": 4, "n_audit_reserved": 0},
    )


def test_pareto_front_identifies_non_dominated():
    df = pd.DataFrame({
        "condition_id": ["c1", "c2", "c3"],
        "f1_micro_mean": [0.70, 0.75, 0.65],
        "pipeline_incremental_mean": [1.0, 2.0, 1.5],
    })
    pf = pareto_front(df, "f1_micro_mean", "pipeline_incremental_mean")
    assert list(pf["condition_id"]) == ["c1", "c2"]


def test_assemble_and_validate_study_result_with_operator_diagnostics():
    dataset = _make_dummy_dataset()
    plan = build_experiment_plan(persist_registry=False)
    raw_metrics = pd.DataFrame([
        {
            "family": "core_factorial",
            "features": "minilm",
            "fontes": "minilm",
            "n_fontes": 1,
            "input_dim": 384,
            "modelo": "mlp",
            "fold": 1,
            "seed": 42,
            "f1_micro": 0.72,
            "f1_macro": 0.68,
            "precision_micro": 0.75,
            "recall_micro": 0.70,
            "train_seconds": 0.10,
            "predict_seconds": 0.01,
            "pipeline_seconds_incremental": 0.11,
        }
    ])
    raw_class = pd.DataFrame([
        {
            "family": "core_factorial",
            "features": "minilm",
            "modelo": "mlp",
            "fold": 1,
            "seed": 42,
            "class_name": "Ação",
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "support": 2,
        }
    ])
    exp_result = standardize_experiment_result(raw_metrics, raw_class, plan)
    diagnostics = {
        "graph_operator_comparison": pd.DataFrame([
            {
                "condition_id": "cond_1",
                "condition_label": "MLP",
                "f1_micro_mean": 0.72,
                "f1_macro_mean": 0.68,
                "f1_micro_sd": 0.01,
                "f1_macro_sd": 0.01,
            }
        ])
    }
    study = assemble_study_result(dataset, exp_result, diagnostics=diagnostics)
    assert validate_study_result(study) is True
    profile = build_dataset_profile(dataset)
    assert "dataset_overview" in profile
