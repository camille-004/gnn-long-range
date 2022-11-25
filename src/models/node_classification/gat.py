import torch.nn.functional as F
from torch.nn import ELU, Dropout, LogSoftmax
from torch_geometric.nn import GATv2Conv, Sequential

from ...utils import load_config
from .base import BaseNodeClassifier

config = load_config("model_config.json")
gat_config = config["gat_params"]


class NodeLevelGAT(BaseNodeClassifier):
    """PyTorch Lightning module of GAT for node classification."""

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
            "x, edge_index",
            [
                (Dropout(p=self.dropout), "x -> xd"),
                (
                    GATv2Conv(
                        self.num_features,
                        self.hidden_channels,
                        heads=self.num_heads,
                    ),
                    "xd, edge_index -> x1",
                ),
                (ELU(), "x1 -> x1a"),
                (Dropout(p=self.dropout), "x1a -> x1d"),
                (
                    GATv2Conv(
                        self.hidden_channels * self.num_heads,
                        self.num_classes,
                        heads=1,
                    ),
                    "x1d, edge_index -> x2",
                ),
                (LogSoftmax(), "x2 -> x_out"),
            ],
        )

        self.loss_fn = F.nll_loss
