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

config = load_config("model_config.json")
gin_config = config["gin_params"]


class GraphLevelGINWithCat(BaseGraphClassifier):
    """PyTorch Lightning module of GIN with graph embedding
    concatenation for graph classification."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
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

    def __init__(self, **kwargs) -> None:
        super().__init__()
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

        self.model = Sequential(
            "x, edge_index, batch_index",
            [
                (
                    GINConv(
                        nn.Sequential(
                            Linear(self.num_features, self.hidden_channels),
                            ReLU(),
                            Linear(self.hidden_channels, self.hidden_channels),
                        )
                    ),
                    "x, edge_index -> x1",
                ),
                (ReLU(), "x1 -> x1a"),
                (BatchNorm1d(self.hidden_channels), "x1a -> x1b"),
                (
                    GINConv(
                        nn.Sequential(
                            Linear(self.hidden_channels, self.hidden_channels),
                            ReLU(),
                            Linear(self.hidden_channels, self.hidden_channels),
                        )
                    ),
                    "x1b, edge_index -> x2",
                ),
                (ReLU(), "x2 -> x2a"),
                (BatchNorm1d(self.hidden_channels), "x2a -> x2b"),
                (
                    Linear(self.hidden_channels, self.hidden_channels),
                    "x2b -> x3",
                ),
                (ReLU(), "x3 -> x3a"),
                (Dropout(p=self.dropout), "x3a -> x3d"),  # Should be p=0.5
                (
                    Linear(self.hidden_channels, self.hidden_channels),
                    "x3d -> x4",
                ),
                (global_mean_pool, "x4, batch_index -> x5"),
                (Dropout(p=self.dropout), "x5 -> x5d"),
                (
                    Linear(self.hidden_channels, self.num_classes),
                    "x5d -> x_out",
                ),
            ],
        )

        self.loss_fn = F.cross_entropy
