"""Tests for the model-agnostic registry and GraphContext contract."""

import numpy as np
from veritas_ml.core.contracts import GraphContext
from veritas_ml.models import (
    ModelOutput,
    clear_registry,
    get_model_spec,
    get_special_experiments,
    list_models,
    multilabel_metrics,
    register_model,
    register_special_experiment,
    run_model,
)


def test_registry_starts_clean_and_registers_notebook_models():
    clear_registry()
    assert list_models() == []

    def dummy_adapter(x_train, y_train, x_eval, graph_context, seed):
        prob = np.zeros((len(x_eval), y_train.shape[1]), dtype=np.float32)
        return prob, 64, "parameters", 0.01, 0.002

    spec = register_model(
        "notebook_mlp",
        dummy_adapter,
        display_name="Notebook MLP",
        family="neural",
        role="individual_baseline",
        seeds=(10, 20),
        training_config={"epochs": 50, "lr": 1e-3},
    )
    assert spec.model_id == "notebook_mlp"
    assert list_models(enabled_only=True) == ["notebook_mlp"]
    assert get_model_spec("notebook_mlp").training_config["epochs"] == 50

    register_special_experiment({
        "experiment_id": "special_op_1",
        "model_id": "notebook_mlp",
        "features": "minilm",
        "family": "custom_special",
    })
    assert len(get_special_experiments()) == 1


def test_custom_notebook_model_registration_and_execution():
    clear_registry()

    def dummy_tf_like_adapter(x_train, y_train, x_eval, graph_context, seed):
        assert graph_context is not None
        assert graph_context.p_eval is not None
        rng = np.random.default_rng(seed)
        prob = rng.uniform(0.0, 1.0, size=(len(x_eval), y_train.shape[1])).astype(np.float32)
        return prob, 128, "parameters", 0.05, 0.01

    register_model(
        "custom_tf_gnn",
        dummy_tf_like_adapter,
        display_name="Custom TF GNN",
        family="graph",
        role="relational",
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
