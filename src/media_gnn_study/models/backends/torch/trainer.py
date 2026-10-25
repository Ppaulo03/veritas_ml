"""Training and inference loops for PyTorch models."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from ....core.config import (
    DEVICE,
    EPOCHS,
    HETERO_TERM_CFG,
    LR,
    WEIGHT_DECAY,
)
from ...base import ModelOutput
from .modules import (
    HeteroTermGATv2,
    HeteroTermGNN,
    HomogeneousPyGClassifier,
    LinearClassifier,
    MLPGraphClassifier,
)


def _term_to_work_edges(
    row_incidence: np.ndarray,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a row-normalized work-term matrix to PyG directed bipartite edges and edge attributes."""
    arr = np.asarray(row_incidence, dtype=np.float32)
    work_idx, term_idx = np.nonzero(arr > 0)
    if len(work_idx) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float32, device=device)
    else:
        edge_index = torch.tensor(np.vstack([term_idx, work_idx]), dtype=torch.long, device=device)
        edge_attr = torch.tensor(arr[work_idx, term_idx, None], dtype=torch.float32, device=device)
    return edge_index, edge_attr


def fit_predict_neural(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    model: str,
    seed: int,
    p_train: np.ndarray | torch.Tensor | None = None,
    p_eval: np.ndarray | torch.Tensor | None = None,
    device: torch.device = DEVICE,
) -> ModelOutput:
    """Train and evaluate Linear, MLP, or Fixed-kNN GNN models."""
    torch.manual_seed(seed)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xe = torch.tensor(np.concatenate([x_train, x_eval]), dtype=torch.float32, device=device)

    pt = torch.as_tensor(p_train, dtype=torch.float32, device=device) if p_train is not None else None
    pe = torch.as_tensor(p_eval, dtype=torch.float32, device=device) if p_eval is not None else None

    if model == "linear":
        net = LinearClassifier(xt.shape[1], yt.shape[1]).to(device)
        pt = pe = None
    elif model in ("mlp", "gnn"):
        net = MLPGraphClassifier(xt.shape[1], yt.shape[1]).to(device)
        if model != "gnn":
            pt = pe = None
    else:
        raise ValueError(f"Unknown neural model: {model}")

    positives = yt.sum(dim=0)
    if (positives == 0).any() or (positives == len(yt)).any():
        raise ValueError("Degenerate class distribution in training fold.")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(yt) - positives) / positives)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net(xt, pt), yt)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss diverged for model: {model}")
        loss.backward()
        opt.step()
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    net.eval()
    with torch.inference_mode():
        logits = net(xe, pe)[len(xt):]
        prob = torch.sigmoid(logits).cpu().numpy()
    pred_s = time.perf_counter() - t1

    param_count = sum(p.numel() for p in net.parameters())
    return ModelOutput(prob, param_count, "parameters", train_s, pred_s)


def fit_predict_homogeneous_operator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    model: str,
    seed: int,
    edge_train: torch.Tensor,
    edge_eval: torch.Tensor,
    device: torch.device = DEVICE,
) -> ModelOutput:
    """Train and evaluate homogeneous PyG operator models (GCN, GraphSAGE, GATv2)."""
    torch.manual_seed(seed)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xe = torch.tensor(np.concatenate([x_train, x_eval]), dtype=torch.float32, device=device)

    net = HomogeneousPyGClassifier(xt.shape[1], yt.shape[1], model).to(device)
    positives = yt.sum(dim=0)
    if (positives == 0).any() or (positives == len(yt)).any():
        raise ValueError(f"Degenerate class distribution for homogeneous {model}.")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(yt) - positives) / positives)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net(xt, edge_train), yt)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss diverged for {model}")
        loss.backward()
        opt.step()
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    net.eval()
    with torch.inference_mode():
        logits = net(xe, edge_eval)[len(xt):]
        prob = torch.sigmoid(logits).cpu().numpy()
    pred_s = time.perf_counter() - t1

    param_count = sum(p.numel() for p in net.parameters())
    return ModelOutput(prob, param_count, "parameters", train_s, pred_s)


def fit_predict_hetero_term(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    row_train: np.ndarray,
    col_train: np.ndarray,
    row_eval: np.ndarray,
    seed: int,
    device: torch.device = DEVICE,
) -> ModelOutput:
    """Train and evaluate bipartite work-term HeteroTermGNN."""
    torch.manual_seed(seed)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xe = torch.tensor(x_eval, dtype=torch.float32, device=device)
    rt = torch.tensor(row_train, dtype=torch.float32, device=device)
    ct = torch.tensor(col_train, dtype=torch.float32, device=device)
    re = torch.tensor(row_eval, dtype=torch.float32, device=device)

    net = HeteroTermGNN(
        xt.shape[1],
        yt.shape[1],
        hidden=HETERO_TERM_CFG["hidden"],
        dropout=HETERO_TERM_CFG["dropout"],
    ).to(device)

    positives = yt.sum(dim=0)
    if (positives == 0).any() or (positives == len(yt)).any():
        raise ValueError("Degenerate class distribution in hetero term train.")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(yt) - positives) / positives)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net.forward_train(xt, rt, ct), yt)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss diverged in hetero term GNN")
        loss.backward()
        opt.step()
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    net.eval()
    with torch.inference_mode():
        logits = net.forward_eval(xt, xe, rt, ct, re)
        prob = torch.sigmoid(logits).cpu().numpy()
    pred_s = time.perf_counter() - t1

    param_count = sum(p.numel() for p in net.parameters())
    return ModelOutput(prob, param_count, "parameters", train_s, pred_s)


def fit_predict_hetero_term_gatv2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    row_train: np.ndarray,
    col_train: np.ndarray,
    row_eval: np.ndarray,
    seed: int,
    device: torch.device = DEVICE,
) -> ModelOutput:
    """Train and evaluate bipartite work-term HeteroTermGATv2 with learned attention."""
    torch.manual_seed(seed)
    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xe = torch.tensor(x_eval, dtype=torch.float32, device=device)
    ct = torch.tensor(col_train, dtype=torch.float32, device=device)

    edge_tr, attr_tr = _term_to_work_edges(row_train, device=device)
    edge_ev, attr_ev = _term_to_work_edges(row_eval, device=device)

    net = HeteroTermGATv2(
        xt.shape[1],
        yt.shape[1],
        hidden=HETERO_TERM_CFG["hidden"],
        dropout=HETERO_TERM_CFG["dropout"],
    ).to(device)

    positives = yt.sum(dim=0)
    if (positives == 0).any() or (positives == len(yt)).any():
        raise ValueError("Degenerate class distribution in hetero term GATv2 train.")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(yt) - positives) / positives)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net.forward_train(xt, ct, edge_tr, attr_tr), yt)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss diverged in hetero term GATv2")
        loss.backward()
        opt.step()
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    net.eval()
    with torch.inference_mode():
        logits = net.forward_eval(xt, xe, ct, edge_ev, attr_ev)
        prob = torch.sigmoid(logits).cpu().numpy()
    pred_s = time.perf_counter() - t1

    param_count = sum(p.numel() for p in net.parameters())
    return ModelOutput(prob, param_count, "parameters", train_s, pred_s)
