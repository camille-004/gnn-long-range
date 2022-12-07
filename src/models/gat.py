from typing import Optional

import torch.nn as nn
from torch_geometric.nn import GATConv

from .base import BaseNodeClassifier


class NodeLevelGAT(BaseNodeClassifier):
    """PyTorch Lightning module of GAT for node classification."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int,
        n_hidden: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        n_heads: int,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super(NodeLevelGAT, self).__init__(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
        )

        if self.activation is None:
            self.activation = nn.ELU()

        self._model_name = "node_GAT"
        self.convs = nn.ModuleList()

        if n_hidden > 0:
            self.convs.append(GATConv(num_features, hidden_dim, n_heads))
        else:
            self.convs.append(
                GATConv(num_features, hidden_dim * n_heads, n_heads)
            )

        for _ in range(n_hidden):
            self.convs.append(
                GATConv(hidden_dim * n_heads, hidden_dim, n_heads)
            )

        self.convs.append(
            GATConv(hidden_dim * n_heads, num_classes, 1, concat=False)
        )
