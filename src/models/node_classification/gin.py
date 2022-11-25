import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Linear, LogSoftmax, ReLU
from torch_geometric.nn import GINConv, Sequential

from src.utils import load_config

from .base import BaseNodeClassifier

config = load_config("model_config.json")
gin_config = config["gin_params"]


class NodeLevelGIN(BaseNodeClassifier):
    """PyTorch Lightning module of GIN for graph classification."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
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

        self.model = Sequential(
            "x, edge_index",
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
                (Dropout(p=self.dropout), "x3a -> x3d"),
                (Linear(self.hidden_channels, self.num_classes), "x3d -> x4"),
                (LogSoftmax(dim=-1), "x4 -> x_out"),
            ],
        )

        self.loss_fn = F.cross_entropy
