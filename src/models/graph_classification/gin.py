from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, LogSoftmax, ReLU
from torch_geometric.nn import JumpingKnowledge

# isort: off
from torch_geometric.nn import (  # noqa
    GINConv,
    Sequential,
    global_add_pool,
    global_mean_pool,
)

# isort: on

from src.utils import load_config

from .base import BaseGraphClassifier

config = load_config("model_config.yaml")
gin_config = config["gin_params"]


class GraphLevelGIN(BaseGraphClassifier):
    """PyTorch Lightning module of GIN for graph classification."""

    def __init__(
        self, n_hidden: int, activation: nn.Module = None, **kwargs
    ) -> None:
        super().__init__(n_hidden, activation)

        if self.activation is None:
            self.activation = ReLU()

        self._model_name = "graph_GIN"

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

        hidden = []

        for i in range(1, self.n_hidden + 1):
            hidden.append((Dropout(p=self.dropout), f"x{i} -> x{i}d"))
            hidden.append(
                (
                    GINConv(
                        nn.Sequential(
                            Linear(self.hidden_channels, self.hidden_channels),
                            self.activation,
                            Linear(self.hidden_channels, self.hidden_channels),
                            self.activation,
                            BatchNorm1d(self.hidden_channels),
                        )
                    ),
                    f"x{i}d, edge_index -> x{i + 1}",
                )
            )

        self.model = Sequential(
            "x, edge_index, batch_index",
            [
                (Dropout(p=self.dropout), "x -> xd"),
                (
                    GINConv(
                        nn.Sequential(
                            Linear(self.num_features, self.hidden_channels),
                            self.activation,
                            Linear(self.hidden_channels, self.hidden_channels),
                            self.activation,
                            BatchNorm1d(self.hidden_channels),
                        )
                    ),
                    "xd, edge_index -> x1",
                ),
                *hidden,
                (
                    global_mean_pool,
                    f"x{n_hidden}, batch_index -> x{n_hidden + 2}",
                ),
                (
                    Linear(self.hidden_channels, self.hidden_channels),
                    f"x{n_hidden + 2} -> x{n_hidden + 3}",
                ),
                (self.activation, f"x{n_hidden + 3} -> x{n_hidden + 3}a"),
                (
                    Dropout(p=self.dropout),
                    f"x{n_hidden + 3}a -> x{n_hidden + 3}d",
                ),
                (
                    Linear(self.hidden_channels, self.num_classes),
                    f"x{n_hidden + 3}d -> x{n_hidden + 4}",
                ),
                (LogSoftmax(), f"x{n_hidden + 4} -> x_out"),
            ],
        )

        print(self.model)

        self.loss_fn = F.cross_entropy


class GraphLeveLGINWithJK(BaseGraphClassifier):
    def __init__(self, n_hidden: int, activation: nn.Module = None, **kwargs):
        super().__init__(n_hidden, activation)

        if self.activation is None:
            self.activation = ReLU()

        self._model_name = "graph_GIN"

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

        if "jk_mode" in kwargs.keys():
            assert kwargs["jk_mode"] in ["cat", "max", "lstm"]
            self.mode = kwargs["jk_mode"]
        else:
            self.mode = "max"

        self.conv_1 = GINConv(
            nn.Sequential(
                Linear(self.num_features, self.hidden_channels),
                self.activation,
                Linear(self.hidden_channels, self.hidden_channels),
                self.activation,
                BatchNorm1d(self.hidden_channels),
            )
        )
        self.hidden = nn.ModuleList()

        for _ in range(n_hidden):
            self.hidden.append(
                GINConv(
                    nn.Sequential(
                        Linear(self.hidden_channels, self.hidden_channels),
                        self.activation,
                        Linear(self.hidden_channels, self.hidden_channels),
                        self.activation,
                        BatchNorm1d(self.hidden_channels),
                    )
                )
            )

        self.jk = JumpingKnowledge(self.mode)

        if self.mode == "cat":
            self.lin_1 = Linear(
                self.hidden_channels * self.n_hidden, self.hidden_channels
            )
        else:
            self.lin_1 = Linear(self.hidden_channels, self.hidden_channels)

        self.lin_2 = Linear(self.hidden_channels, self.num_classes)

    def forward(self, x: Any, edge_index: Any, batch_index: Any) -> Tensor:
        x = self.conv_1(x, edge_index)
        xs = []

        for conv in self.hidden:
            x = conv(x, edge_index)
            xs += [x]

        x = self.jk(xs)
        x = global_mean_pool(x, batch_index)
        x = self.activation(self.lin_1(x))
        x = F.dropout(x, p=self.dropout)
        x = self.lin_2(x)

        return F.log_softmax(x, dim=-1)
