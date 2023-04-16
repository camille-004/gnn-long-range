import warnings
from typing import Optional

import torch.nn as nn
from .sognn_layer import SOGNNConv

from .base import BaseNodeClassifier


class NodeLevelSOGNN(BaseNodeClassifier):
    """PyTorch Lightning module of GCN for node classification."""

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
        super(NodeLevelSOGNN, self).__init__(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            activation=activation
        )

        self._model_name = "node_SOGNN"

        if self.activation is None:
            self.activation = nn.ReLU()
            warnings.warn(
                "Using ReLU activation function. Non-differentiable"
                "activation may yield inaccurate influences."
            )

        self.convs = nn.ModuleList()
        self.convs.append(SOGNNConv(num_features, hidden_dim))

        for _ in range(n_hidden):
            self.convs.append(SOGNNConv(hidden_dim, hidden_dim))

        self.convs.append(SOGNNConv(hidden_dim, num_classes))
