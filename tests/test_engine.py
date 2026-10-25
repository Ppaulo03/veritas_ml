"""Tests for the experimental plan and result standardization in engine."""

import numpy as np
import pandas as pd
from veritas_ml.engine import (
    BUNDLE_REGISTRY,
    build_experiment_plan,
    plan_summary,
    standardize_experiment_result,
)
from veritas_ml.models import clear_registry, register_model


def _register_test_models():
    clear_registry()

    def _noop(x_tr, y_tr, x_ev, g_ctx, seed):
        return np.zeros((len(x_ev), y_tr.shape[1]), dtype=np.float32), 10, "parameters", 0.01, 0.01

    register_model("mlp", _noop, display_name="MLP", family="neural", role="individual_baseline")
    register_model("gnn", _noop, display_name="GNN", family="graph", role="relational")


def test_build_experiment_plan_and_summary():
    _register_test_models()
    plan = build_experiment_plan(persist_registry=False)
    assert len(plan.registry) == len(BUNDLE_REGISTRY)
    assert len(plan.registry) == 55
    assert plan.model_ids == ("mlp", "gnn")
    summary = plan_summary(plan)
    assert summary["bundles"] == 55
    assert summary["main_models"] == 2
    assert summary["conditions"] == summary["main_conditions"] + summary["special_conditions"]


def test_standardize_experiment_result_builds_observations():
    _register_test_models()
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
    assert res.conditions.iloc[0].model_name == "MLP"
    assert set(res.metric_observations["metric"]) == {
        "f1_micro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
    }
    assert len(res.class_metric_observations) == 4
