import warnings
from typing import Any, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv

from src.utils import dirichlet_energy, get_graph_laplacian, load_config

from .base import BaseNodeClassifier

config = load_config("model_config.yaml")
gin_config = config["gin_params"]


class NodeLevelGIN(BaseNodeClassifier):
    """PyTorch Lightning module of GIN for graph classification."""

    def __init__(
        self, n_hidden: int, activation: nn.Module = None, **kwargs
    ) -> None:
        super().__init__(n_hidden, activation)

        if activation is None:
            self.activation = ReLU()
            warnings.warn(
                "Using ReLU activation function. Non-differentiable"
                "activation may yield inaccurate influences."
            )

        self._model_name = "node_GIN"

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
            else gin_config["dropout"]
        )
        self.lr: float = (
            kwargs["lr"] if "lr" in kwargs.keys() else gin_config["lr"]
        )
        self.weight_decay: float = (
            kwargs["weight_decay"]
            if "weight_decay" in kwargs.keys()
            else gin_config["weight_decay"]
        )

        self.energies = None

        self.conv_in = GINConv(
            Sequential(
                Linear(self.num_features, self.hidden_channels),
                self.activation,
                Linear(self.hidden_channels, self.hidden_channels),
                self.activation,
                BatchNorm1d(self.hidden_channels),
            ),
            train_eps=True,
        )

        self.convs = nn.ModuleList()

        for _ in range(n_hidden):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(self.hidden_channels, self.hidden_channels),
                        self.activation,
                        Linear(self.hidden_channels, self.hidden_channels),
                        self.activation,
                        BatchNorm1d(self.hidden_channels),
                    ),
                    train_eps=True,
                )
            )

        self.lin_1 = Linear(self.hidden_channels, self.hidden_channels)
        self.lin_2 = Linear(self.hidden_channels, self.num_classes)

        self.loss_fn = F.cross_entropy

    def forward(self, x: Any, edge_index: Any) -> Tuple[Tensor, Tensor]:
        """GIN forward pass."""
        _L = get_graph_laplacian(edge_index, x.size(0))
        self.energies = []

        x = self.conv_in(x, edge_index)
        energy = dirichlet_energy(x, _L)
        self.energies.append(energy)

        for i in range(self.n_hidden):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
            energy = dirichlet_energy(x, _L)
            self.energies.append(energy)

        return F.log_softmax(x, dim=1), x
