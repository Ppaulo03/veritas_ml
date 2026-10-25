"""Classical baseline models: Random Forest and XGBoost."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ...core.config import CPU_THREADS
from ..base import ModelOutput

CLASSICAL_N_JOBS = CPU_THREADS

RF_CFG: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": CLASSICAL_N_JOBS,
}

XGB_CFG: dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": CLASSICAL_N_JOBS,
    "verbosity": 0,
}


def _positive_prob_from_rf(model: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    """Extract probability of class 1 for each label from multilabel RandomForestClassifier."""
    prob_list = model.predict_proba(x)
    cols: list[np.ndarray] = []
    for j, p in enumerate(prob_list):
        classes = np.asarray(model.classes_[j])
        if 1 in classes:
            idx = int(np.where(classes == 1)[0][0])
            cols.append(p[:, idx])
        else:
            cols.append(np.zeros(len(x), dtype=np.float32))
    return np.column_stack(cols).astype(np.float32)


def fit_predict_classical(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    model: str,
    seed: int,
) -> ModelOutput:
    """Train and evaluate classical tree/boosting models."""
    if model == "rf":
        clf = RandomForestClassifier(random_state=seed, **RF_CFG)
        t0 = time.perf_counter()
        clf.fit(x_train, y_train)
        train_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        prob = _positive_prob_from_rf(clf, x_eval)
        pred_s = time.perf_counter() - t1

        complexity = int(sum(tree.tree_.node_count for tree in clf.estimators_))
        return ModelOutput(prob, complexity, "tree_nodes", train_s, pred_s)

    if model == "xgb":
        import xgboost as xgb

        estimators: list[xgb.XGBClassifier] = []
        t0 = time.perf_counter()
        for j in range(y_train.shape[1]):
            yj = y_train[:, j].astype(int)
            pos = max(1, int(yj.sum()))
            neg = max(1, len(yj) - pos)
            clf = xgb.XGBClassifier(
                random_state=seed + j,
                scale_pos_weight=neg / pos,
                **XGB_CFG,
            )
            clf.fit(x_train, yj)
            estimators.append(clf)
        train_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        prob = np.column_stack([
            clf.predict_proba(x_eval)[:, 1] for clf in estimators
        ]).astype(np.float32)
        pred_s = time.perf_counter() - t1

        complexity = int(sum(clf.get_booster().num_boosted_rounds() for clf in estimators))
        return ModelOutput(prob, complexity, "boosted_trees", train_s, pred_s)

    raise ValueError(f"Unknown classical model: {model}")
