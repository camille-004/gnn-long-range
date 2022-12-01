import warnings
from typing import Any, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ReLU
from torch_geometric.nn import GCNConv

from src.utils import (
    dirichlet_energy,
    get_graph_laplacian,
    load_config,
    rayleigh_quotient,
)

from .base import BaseNodeClassifier

config = load_config("model_config.yaml")
gcn_config = config["gcn_params"]


class NodeLevelGCN(BaseNodeClassifier):
    """PyTorch Lightning module of GCN for node classification."""

    def __init__(
        self, n_hidden, activation: nn.Module = None, **kwargs
    ) -> None:
        super().__init__(n_hidden, activation)

        if activation is None:
            self.activation = ReLU()
            warnings.warn(
                "Using ReLU activation function. Non-differentiable"
                "activation may yield inaccurate influences."
            )

        self._model_name = "node_GCN"

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
            else gcn_config["node"]["hidden_channels"]
        )
        self.dropout: int = (
            kwargs["dropout"]
            if "dropout" in kwargs.keys()
            else gcn_config["node"]["dropout"]
        )
        self.lr: float = (
            kwargs["lr"] if "lr" in kwargs.keys() else gcn_config["node"]["lr"]
        )
        self.weight_decay: float = (
            kwargs["weight_decay"]
            if "weight_decay" in kwargs.keys()
            else gcn_config["node"]["weight_decay"]
        )

        self.conv_in = GCNConv(self.num_features, self.hidden_channels)
        self.hidden = nn.ModuleList([])

        for _ in range(n_hidden):
            self.hidden.append(
                GCNConv(self.hidden_channels, self.hidden_channels)
            )

        self.conv_out = GCNConv(self.hidden_channels, self.num_classes)

        self.loss_fn = F.cross_entropy
        self.energies = None
        self.rayleigh = None

    def forward(self, x: Any, edge_index: Any) -> Tuple[Tensor, Tensor]:
        """Node GCN forward pass."""
        self.energies = []
        self.rayleigh = []

        _L = get_graph_laplacian(edge_index, x.size(0))
        h = self.conv_in(x, edge_index)

        # Calculate energy of input layer
        self.energies.append(dirichlet_energy(h, _L))
        self.rayleigh.append(rayleigh_quotient(h, _L))

        h = self.activation(h)

        for i in range(self.n_hidden):
            h = self.hidden[i](h, edge_index)

            # Calculate energy of hidden layers
            self.energies.append(dirichlet_energy(h, _L))
            self.rayleigh.append(rayleigh_quotient(h, _L))

            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv_out(h, edge_index)
        h = self.activation(h)

        # Calculate energy of output layer
        self.energies.append(dirichlet_energy(h, _L))
        self.rayleigh.append(rayleigh_quotient(h, _L))

        return F.log_softmax(h, dim=1), h
