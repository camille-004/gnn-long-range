import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear, LogSoftmax, ReLU

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


class GraphLevelGINWithCat(BaseGraphClassifier):
    """PyTorch Lightning module of GIN with graph embedding
    concatenation for graph classification."""

    def __init__(
        self, n_hidden: int, activation: nn.Module = ReLU(), **kwargs
    ) -> None:
        super().__init__(n_hidden, activation)
        self._model_name = "graph_GIN_cat"

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

        self.model = Sequential(
            "x, edge_index, batch_index",
            [
                (
                    GINConv(
                        nn.Sequential(
                            Linear(self.num_features, self.hidden_channels),
                            BatchNorm1d(self.hidden_channels),
                            ReLU(),
                            Linear(self.hidden_channels, self.hidden_channels),
                            ReLU(),
                        ),
                    ),
                    "x, edge_index -> x1",
                ),
                (
                    GINConv(
                        nn.Sequential(
                            Linear(self.hidden_channels, self.hidden_channels),
                            BatchNorm1d(self.hidden_channels),
                            ReLU(),
                            Linear(self.hidden_channels, self.hidden_channels),
                            ReLU(),
                        )
                    ),
                    "x1, edge_index -> x2",
                ),
                (
                    GINConv(
                        nn.Sequential(
                            Linear(self.hidden_channels, self.hidden_channels),
                            BatchNorm1d(self.hidden_channels),
                            ReLU(),
                            Linear(self.hidden_channels, self.hidden_channels),
                            ReLU(),
                        )
                    ),
                    "x2, edge_index -> x3",
                ),
                (global_add_pool, "x1, batch_index -> x1"),
                (global_add_pool, "x2, batch_index -> x2"),
                (global_add_pool, "x3, batch_index -> x3"),
                (
                    lambda x1, x2, x3: torch.cat((x1, x2, x3), dim=1),
                    "x1, x2, x3 -> x",
                ),
                (Linear(self.hidden_channels * 3, self.hidden_channels * 3)),
                (ReLU()),
                (Dropout(p=self.dropout)),
                (Linear(self.hidden_channels * 3, self.num_classes)),
                (LogSoftmax(), "x -> x_out"),
            ],
        )

        self.loss_fn = F.cross_entropy


class GraphLevelGIN(BaseGraphClassifier):
    """PyTorch Lightning module of GIN for graph classification."""

    def __init__(
        self, n_hidden: int, activation: nn.Module = ReLU(), **kwargs
    ) -> None:
        super().__init__(n_hidden, activation)
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

        self.loss_fn = F.cross_entropy
