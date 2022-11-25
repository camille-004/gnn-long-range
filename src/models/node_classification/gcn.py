import torch.nn.functional as F
from torch.nn import Dropout, ReLU
from torch_geometric.nn import GCNConv, Sequential

from src.utils import load_config

from .base import BaseNodeClassifier

config = load_config("model_config.yaml")
gcn_config = config["gcn_params"]


class NodeLevelGCN(BaseNodeClassifier):
    """PyTorch Lightning module of GCN for node classification."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
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

        self.model = Sequential(
            "x, edge_index",
            [
                (
                    GCNConv(self.num_features, self.hidden_channels),
                    "x, edge_index -> x1",
                ),
                (ReLU(), "x1 -> x1a"),
                (Dropout(p=self.dropout), "x1a -> x1d"),
                (
                    GCNConv(self.hidden_channels, self.num_classes),
                    "x1d, edge_index -> x_out",
                ),
            ],
        )

        self.loss_fn = F.cross_entropy
