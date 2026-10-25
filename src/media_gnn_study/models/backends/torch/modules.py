"""PyTorch and PyTorch Geometric neural network architectures."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ....core.config import DROPOUT, HIDDEN


class LinearClassifier(nn.Module):
    """Simple linear multilabel classifier."""

    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_labels)

    def forward(self, x: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(x)


class MLPGraphClassifier(nn.Module):
    """Two-layer MLP classifier supporting fixed-weight message propagation."""

    def __init__(self, in_dim: int, n_labels: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_labels)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _propagate(p: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
        if p is None:
            return x
        return torch.sparse.mm(p, x) if p.is_sparse else p @ x

    def forward(self, x: torch.Tensor, p: torch.Tensor | None = None) -> torch.Tensor:
        h = self._propagate(p, x)
        h = self.dropout(F.relu(self.fc1(h)))
        h = self._propagate(p, h)
        return self.fc2(h)


class HomogeneousPyGClassifier(nn.Module):
    """Homogeneous GNN using PyTorch Geometric operators (GCN, GraphSAGE, GATv2)."""

    def __init__(
        self,
        in_dim: int,
        n_labels: int,
        operator: str,
        hidden: int = HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv

        self.operator = str(operator)
        self.dropout = nn.Dropout(dropout)

        if self.operator == "gcn":
            self.conv1 = GCNConv(in_dim, hidden, add_self_loops=True, normalize=True)
            self.conv2 = GCNConv(hidden, n_labels, add_self_loops=True, normalize=True)
        elif self.operator == "graphsage":
            self.conv1 = SAGEConv(in_dim, hidden, aggr="mean", root_weight=True)
            self.conv2 = SAGEConv(hidden, n_labels, aggr="mean", root_weight=True)
        elif self.operator == "gatv2":
            self.conv1 = GATv2Conv(
                in_dim, hidden, heads=1, concat=False, dropout=dropout, add_self_loops=True
            )
            self.conv2 = GATv2Conv(
                hidden, n_labels, heads=1, concat=False, dropout=dropout, add_self_loops=True
            )
        else:
            raise ValueError(f"Unsupported PyG operator: {operator}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.dropout(F.relu(h))
        return self.conv2(h, edge_index)


class HeteroTermGNN(nn.Module):
    """Bipartite work↔term GNN for classifying work nodes.

    Term node states are aggregated exclusively from training works;
    validation works only query those states, preventing validation→term→validation leakage.
    """

    def __init__(
        self,
        work_dim: int,
        n_labels: int,
        hidden: int = HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.work_encoder = nn.Linear(work_dim, hidden)
        self.work_to_term = nn.Linear(hidden, hidden, bias=False)
        self.self_path = nn.Linear(hidden, hidden, bias=False)
        self.term_to_work = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_labels)

    @staticmethod
    def _term_state(h_train: torch.Tensor, col_normalized_train: torch.Tensor) -> torch.Tensor:
        return col_normalized_train.T @ h_train

    def _work_logits(
        self,
        h_work: torch.Tensor,
        row_normalized_incidence: torch.Tensor,
        term_state: torch.Tensor,
    ) -> torch.Tensor:
        msg = row_normalized_incidence @ self.term_to_work(term_state)
        h = F.relu(self.self_path(h_work) + msg)
        h = self.dropout(h)
        return self.out(h)

    def forward_train(
        self,
        x_train: torch.Tensor,
        row_train: torch.Tensor,
        col_train: torch.Tensor,
    ) -> torch.Tensor:
        h_train = F.relu(self.work_encoder(x_train))
        term_state = self._term_state(self.work_to_term(h_train), col_train)
        return self._work_logits(h_train, row_train, term_state)

    def forward_eval(
        self,
        x_train: torch.Tensor,
        x_eval: torch.Tensor,
        row_train: torch.Tensor,
        col_train: torch.Tensor,
        row_eval: torch.Tensor,
    ) -> torch.Tensor:
        h_train = F.relu(self.work_encoder(x_train))
        h_eval = F.relu(self.work_encoder(x_eval))
        term_state = self._term_state(self.work_to_term(h_train), col_train)
        return self._work_logits(h_eval, row_eval, term_state)


class HeteroTermGATv2(nn.Module):
    """Bipartite work↔term GNN with learned attention on term→work message aggregation."""

    def __init__(
        self,
        work_dim: int,
        n_labels: int,
        hidden: int = HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        self.work_encoder = nn.Linear(work_dim, hidden)
        self.work_to_term = nn.Linear(hidden, hidden, bias=False)
        self.self_path = nn.Linear(hidden, hidden, bias=False)
        self.term_attention = GATv2Conv(
            (hidden, hidden),
            hidden,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
            edge_dim=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_labels)

    def _term_state(self, h_train: torch.Tensor, col_train: torch.Tensor) -> torch.Tensor:
        return col_train.T @ self.work_to_term(h_train)

    def _logits(
        self,
        h_work: torch.Tensor,
        term_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        msg = self.term_attention((term_state, h_work), edge_index, edge_attr=edge_attr)
        h = self.dropout(F.relu(self.self_path(h_work) + msg))
        return self.out(h)

    def forward_train(
        self,
        x_train: torch.Tensor,
        col_train: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h_train = F.relu(self.work_encoder(x_train))
        term_state = self._term_state(h_train, col_train)
        return self._logits(h_train, term_state, edge_index, edge_attr)

    def forward_eval(
        self,
        x_train: torch.Tensor,
        x_eval: torch.Tensor,
        col_train: torch.Tensor,
        edge_index_eval: torch.Tensor,
        edge_attr_eval: torch.Tensor,
    ) -> torch.Tensor:
        h_train = F.relu(self.work_encoder(x_train))
        h_eval = F.relu(self.work_encoder(x_eval))
        term_state = self._term_state(h_train, col_train)
        return self._logits(h_eval, term_state, edge_index_eval, edge_attr_eval)
