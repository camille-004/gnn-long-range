import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ELU, Dropout, Linear
from torch_geometric.nn import GATConv, Sequential, global_mean_pool

from src.utils import load_config

from .base import BaseGraphClassifier

config = load_config("model_config.yaml")
gat_config = config["gat_params"]["graph"]


class GraphLevelGAT(BaseGraphClassifier):
    """PyTorch Lightning module of GAT for graph classification."""

    def __init__(
        self, n_hidden: int, activation: nn.Module = ELU(), **kwargs
    ) -> None:
        super().__init__(n_hidden, activation)
        self._model_name = "graph_GAT"

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

        hidden = []

        for i in range(1, self.n_hidden + 1):
            hidden.append((Dropout(p=self.dropout), f"x{i}a -> x{i}d"))
            if i == self.n_hidden and self.num_heads > 1:
                hidden.append(
                    (
                        GATConv(
                            self.hidden_channels * self.num_heads,
                            self.hidden_channels,
                            concat=False,
                            dropout=self.dropout,
                        ),
                        f"x{i}d, edge_index -> x{i + 1}",
                    )
                )
            else:
                hidden.append(
                    (
                        GATConv(
                            self.hidden_channels * self.num_heads,
                            self.hidden_channels * self.num_heads,
                            concat=False,
                            heads=self.num_heads,
                            dropout=self.dropout,
                        ),
                        f"x{i}d, edge_index -> x{i + 1}",
                    )
                )
                hidden.append((self.activation, f"x{i + 1} -> x{i + 1}a"))

        if self.num_heads == 1:
            global_mean_pool_args = (
                f"x{n_hidden + 1}a, batch_index -> x{n_hidden + 2}"
            )
        else:
            global_mean_pool_args = (
                f"x{n_hidden + 1}, batch_index -> x{n_hidden + 2}"
            )

        self.model = Sequential(
            "x, edge_index, batch_index",
            [
                (Dropout(p=self.dropout), "x -> xd"),
                (
                    GATConv(
                        self.num_features,
                        self.hidden_channels,
                        heads=self.num_heads,
                        dropout=self.dropout,
                    ),
                    "xd, edge_index -> x1",
                ),
                (self.activation, "x1 -> x1a"),
                *hidden,
                (global_mean_pool, global_mean_pool_args),
                (
                    Dropout(p=self.dropout),
                    f"x{n_hidden + 2} -> x{n_hidden + 2}d",
                ),
                (
                    Linear(self.hidden_channels, self.num_classes),
                    f"x{n_hidden + 2}d -> x_out",
                ),
            ],
        )

        print(self.model)

        self.loss_fn = F.cross_entropy
