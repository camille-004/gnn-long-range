import torch.nn.functional as F
from torch.nn import ELU, Dropout, Linear
from torch_geometric.nn import GATv2Conv, Sequential, global_mean_pool

from ...utils import load_config
from .base import BaseGraphClassifier

config = load_config("model_config.json")
gat_config = config["gat_params"]


class GraphLevelGAT(BaseGraphClassifier):
    """PyTorch Lightning module of GAT for graph classification."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
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

        self.model = Sequential(
            "x, edge_index, batch_index",
            [
                (Dropout(p=self.dropout), "x -> xd"),
                (
                    GATv2Conv(
                        self.num_features,
                        self.hidden_channels,
                        heads=self.num_heads,
                        dropout=self.dropout,
                    ),
                    "xd, edge_index -> x1",
                ),
                (ELU(), "x1 -> x1a"),
                (Dropout(p=self.dropout), "x1a -> x1d"),
                (
                    GATv2Conv(
                        self.hidden_channels * self.num_heads,
                        self.hidden_channels,
                        heads=1,
                        concat=False,
                        dropout=self.dropout,
                    ),
                    "x1d, edge_index -> x2",
                ),
                (global_mean_pool, "x2, batch_index -> x3"),
                (Dropout(p=0.5), "x3 -> x3d"),
                (
                    Linear(self.hidden_channels, self.num_classes),
                    "x3d -> x_out",
                ),
            ],
        )

        self.loss_fn = F.cross_entropy
