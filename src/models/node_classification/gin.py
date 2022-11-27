import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear, LogSoftmax, ReLU
from torch_geometric.nn import GINConv, Sequential

from src.utils import load_config

from .base import BaseNodeClassifier

config = load_config("model_config.yaml")
gin_config = config["gin_params"]


class NodeLevelGIN(BaseNodeClassifier):
    """PyTorch Lightning module of GIN for graph classification."""

    def __init__(self, n_hidden: int, activation: nn.Module = None, **kwargs) -> None:
        super().__init__(n_hidden, activation)

        if self.activation is None:
            self.activation = ReLU()

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

        hidden = []

        for i in range(1, self.n_hidden + 1):
            hidden.append((Dropout(p=self.dropout), f"x{i}b -> x{i}d"))
            hidden.append(
                (
                    GINConv(
                        nn.Sequential(
                            Linear(self.hidden_channels, self.hidden_channels),
                            self.activation,
                            Linear(self.hidden_channels, self.hidden_channels),
                        )
                    ),
                    f"x{i}d, edge_index -> x{i + 1}",
                )
            )
            hidden.append((self.activation, f"x{i + 1} -> x{i + 1}a"))
            hidden.append((BatchNorm1d(self.hidden_channels), f"x{i + 1}a -> x{i + 1}b"))

        self.model = Sequential(
            "x, edge_index",
            [
                (Dropout(p=self.dropout), "x -> xd"),
                (
                    GINConv(
                        nn.Sequential(
                            Linear(self.num_features, self.hidden_channels),
                            self.activation,
                            Linear(self.hidden_channels, self.hidden_channels),
                        )
                    ),
                    "xd, edge_index -> x1",
                ),
                (self.activation, "x1 -> x1a"),
                (BatchNorm1d(self.hidden_channels), "x1a -> x1b"),
                *hidden,
                (
                    Linear(self.hidden_channels, self.hidden_channels),
                    f"x{n_hidden + 1} -> x{n_hidden + 2}",
                ),
                (self.activation, f"x{n_hidden + 2} -> x{n_hidden + 2}a"),
                (
                    Dropout(p=self.dropout),
                    f"x{n_hidden + 2}a -> x{n_hidden + 2}d",
                ),
                (
                    Linear(self.hidden_channels, self.num_classes),
                    f"x{n_hidden + 2}d -> x{n_hidden + 3}",
                ),
                (LogSoftmax(), f"x{n_hidden + 3} -> x_out"),
            ],
        )

        print(self.model)

        self.loss_fn = F.cross_entropy
