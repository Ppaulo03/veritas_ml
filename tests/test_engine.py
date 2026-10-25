"""Tests for the experimental plan and result standardization in engine."""

import pandas as pd
from media_gnn_study.engine import (
    BUNDLE_REGISTRY,
    build_experiment_plan,
    plan_summary,
    standardize_experiment_result,
)


def test_build_experiment_plan_and_summary():
    plan = build_experiment_plan(persist_registry=False)
    assert len(plan.registry) == len(BUNDLE_REGISTRY)
    assert len(plan.registry) == 55
    summary = plan_summary(plan)
    assert summary["bundles"] == 55
    assert summary["conditions"] == summary["main_conditions"] + summary["special_conditions"]


def test_standardize_experiment_result_builds_observations():
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
            "f1_micro": 0.65,
            "f1_macro": 0.55,
            "precision_micro": 0.70,
            "recall_micro": 0.60,
            "train_seconds": 0.12,
            "predict_seconds": 0.01,
            "pipeline_seconds_incremental": 0.13,
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
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "support": 10,
        }
    ])
    res = standardize_experiment_result(raw_metrics, raw_class, plan)
    assert len(res.conditions) == 1
    assert set(res.metric_observations["metric"]) == {
        "f1_micro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
    }
    assert len(res.class_metric_observations) == 4
