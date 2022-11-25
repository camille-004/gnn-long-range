import torch.nn.functional as F
from torch.nn import Dropout, Linear, ReLU
from torch_geometric.nn import GCNConv, Sequential, global_mean_pool

from ...utils import load_config
from .base import BaseGraphClassifier

config = load_config("model_config.json")
gcn_config = config["gcn_params"]


class GraphLevelGCN(BaseGraphClassifier):
    """PyTorch Lightning module of GCN for graph classification."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._model_name = "graph_GCN"

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
            else gcn_config["graph"]["hidden_channels"]
        )
        self.dropout: int = (
            kwargs["dropout"]
            if "dropout" in kwargs.keys()
            else gcn_config["graph"]["dropout"]
        )
        self.lr: float = (
            kwargs["lr"]
            if "lr" in kwargs.keys()
            else gcn_config["graph"]["lr"]
        )
        self.weight_decay: float = (
            kwargs["weight_decay"]
            if "weight_decay" in kwargs.keys()
            else gcn_config["graph"]["weight_decay"]
        )

        self.model = Sequential(
            "x, edge_index, batch_index",
            [
                (
                    GCNConv(self.num_features, self.hidden_channels),
                    "x, edge_index -> x1",
                ),
                (ReLU(), "x1 -> x1a"),
                (
                    GCNConv(self.hidden_channels, self.hidden_channels),
                    "x1a, edge_index -> x2",
                ),
                (ReLU(), "x2 -> x2a"),
                (
                    GCNConv(self.hidden_channels, self.hidden_channels),
                    "x2a, edge_index -> x3",
                ),
                (ReLU(), "x3 -> x3a"),
                (global_mean_pool, "x3a, batch_index -> x4"),
                (Dropout(p=self.dropout), "x4 -> x4d"),
                (
                    Linear(self.hidden_channels, self.num_classes),
                    "x4d -> x_out",
                ),
            ],
        )

        self.loss_fn = F.cross_entropy
