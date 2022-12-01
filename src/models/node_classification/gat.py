from typing import Any, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ELU
from torch_geometric.nn import GATConv

from src.utils import (
    dirichlet_energy,
    get_graph_laplacian,
    load_config,
    rayleigh_quotient,
)

from .base import BaseNodeClassifier

config = load_config("model_config.yaml")
gat_config = config["gat_params"]["node"]


class NodeLevelGAT(BaseNodeClassifier):
    """PyTorch Lightning module of GAT for node classification."""

    def __init__(
        self, n_hidden: int, activation: nn.Module = None, **kwargs
    ) -> None:
        super().__init__(n_hidden, activation)

        if activation is None:
            self.activation = ELU()

        self._model_name = "node_GAT"

        self.num_features: int = (
            kwargs["num_features"]
            if "num_features" in kwargs.keys()
            else config["num_features_default"]
        )
        self.num_classes: int = (
            kwargs["num_classes"]
            if "num_classes" in kwargs.keys()
            else config["num_classes_default"]
        )
        self.hidden_channels: int = (
            kwargs["hidden_channels"]
            if "hidden_channels" in kwargs.keys()
            else config["hidden_channels_default"]
        )
        self.dropout: float = (
            kwargs["dropout"]
            if "dropout" in kwargs.keys()
            else gat_config["dropout"]
        )
        self.num_heads: int = (
            kwargs["num_heads"]
            if "num_heads" in kwargs.keys()
            else gat_config["num_heads"]
        )
        self.lr: float = (
            kwargs["lr"] if "lr" in kwargs.keys() else gat_config["lr"]
        )
        self.weight_decay: float = (
            kwargs["weight_decay"]
            if "weight_decay" in kwargs.keys()
            else gat_config["weight_decay"]
        )

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(self.num_features, self.hidden_channels, self.num_heads)
        )

        for _ in range(self.n_hidden):
            self.convs.append(
                GATConv(
                    self.hidden_channels * self.num_heads,
                    self.hidden_channels,
                    self.num_heads,
                )
            )

        self.convs.append(
            GATConv(
                self.hidden_channels * self.num_heads,
                self.num_classes,
                1,
                concat=False,
            )
        )

        self.loss_fn = F.cross_entropy
        self.energies = None

    def forward(self, x: Any, edge_index: Any) -> Tuple[Tensor, Tensor]:
        """GAT forward pass."""
        _L = get_graph_laplacian(edge_index, x.size(0))
        self.energies = []
        self.rayleigh = []

        for i in range(self.n_hidden + 2):
            x = self.convs[i](x, edge_index)
            energy = dirichlet_energy(x, _L)
            rayleigh = rayleigh_quotient(x, _L)
            x = self.activation(x)
            self.energies.append(energy)
            self.rayleigh.append(rayleigh)

        return F.log_softmax(x, dim=1), x
