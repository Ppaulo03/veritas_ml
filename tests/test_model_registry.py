"""Tests for the framework-agnostic model registry and GraphContext contract."""

import numpy as np
from media_gnn_study.core.contracts import GraphContext
from media_gnn_study.models import (
    ModelOutput,
    list_models,
    multilabel_metrics,
    register_model,
    run_model,
)


def test_builtin_models_are_registered():
    available = list_models()
    for expected in ("linear", "mlp", "gnn", "rf", "xgb", "gcn", "graphsage", "gatv2"):
        assert expected in available


def test_custom_notebook_model_registration_and_execution():
    def dummy_tf_like_adapter(x_train, y_train, x_eval, graph_context, seed):
        assert graph_context is not None
        assert graph_context.p_eval is not None
        rng = np.random.default_rng(seed)
        prob = rng.uniform(0.0, 1.0, size=(len(x_eval), y_train.shape[1])).astype(np.float32)
        return prob, 128, "parameters", 0.05, 0.01

    register_model(
        "custom_tf_gnn",
        dummy_tf_like_adapter,
        metadata={"display_name": "Custom TF GNN", "family": "graph"},
    )

    x_tr = np.ones((4, 8), dtype=np.float32)
    y_tr = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    x_ev = np.ones((2, 8), dtype=np.float32)
    ctx = GraphContext(
        p_train=np.eye(4, dtype=np.float32),
        p_eval=np.eye(6, dtype=np.float32),
    )

    out = run_model("custom_tf_gnn", x_tr, y_tr, x_ev, ctx, seed=42)
    assert isinstance(out, ModelOutput)
    prob, comp, kind, tr_s, pr_s = out
    assert prob.shape == (2, 2)
    assert comp == 128
    assert kind == "parameters"

    metrics = multilabel_metrics(np.array([[1, 0], [0, 1]], dtype=np.float32), prob)
    assert "f1_micro" in metrics and "f1_macro" in metrics
