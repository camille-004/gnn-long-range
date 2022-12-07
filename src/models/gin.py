import warnings
from typing import Any, Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv

from .base import BaseNodeClassifier
from .utils import dirichlet_energy, get_graph_laplacian, rayleigh_quotient


class NodeLevelGIN(BaseNodeClassifier):
    """PyTorch Lightning module of GIN for graph classification."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int,
        n_hidden: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        activation: Optional[nn.Module] = None,
        **kwargs
    ) -> None:
        super(NodeLevelGIN, self).__init__(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            activation=activation
        )

        if self.activation is None:
            self.activation = ReLU()
            warnings.warn(
                "Using ReLU activation function. Non-differentiable"
                "activation may yield inaccurate influences."
            )

        self._model_name = "node_GIN"
        self.convs = nn.ModuleList(
            [
                GINConv(
                    Sequential(
                        Linear(num_features, hidden_dim),
                        self.activation,
                        Linear(hidden_dim, hidden_dim),
                        self.activation,
                        BatchNorm1d(hidden_dim),
                    ),
                    train_eps=True,
                )
            ]
        )

        for _ in range(n_hidden):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        self.activation,
                        Linear(hidden_dim, hidden_dim),
                        self.activation,
                        BatchNorm1d(hidden_dim),
                    ),
                    train_eps=True,
                )
            )

        self.lin_1 = Linear(hidden_dim, hidden_dim)
        self.lin_2 = Linear(hidden_dim, num_classes)

    def forward(self, x: Any, edge_index: Any) -> Tuple[Tensor, Tensor]:
        """GIN forward pass."""
        _L = get_graph_laplacian(edge_index, x.size(0))
        self.energies = []
        self.rayleigh = []

        for i in range(self.n_hidden + 1):
            x = self.convs[i](x, edge_index)
            energy = dirichlet_energy(x, _L)
            rayleigh = rayleigh_quotient(x, _L)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            self.energies.append(energy)
            self.rayleigh.append(rayleigh)

        x = self.activation(self.lin_1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_2(x)

        return F.log_softmax(x, dim=1), x
