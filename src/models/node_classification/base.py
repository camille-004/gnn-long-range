from typing import Any, Callable, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data


class BaseNodeClassifier(pl.LightningModule):
    """Base node classifier."""

    def __init__(self, n_hidden: int, activation: nn.Module, **kwargs) -> None:
        super().__init__()
        pl.utilities.seed.seed_everything(1)

        assert n_hidden >= 0, "Number of hidden layers must be non-negative."
        self.n_hidden = n_hidden
        self.activation = activation

        self._model_name = "base_node_clf"

        self.num_features: int = 0
        self.num_classes: int = 0

        self.loss_fn: Callable[[Tensor, Tensor], Tensor] = F.nll_loss
        self.lr = 0.01
        self.weight_decay = 5e-4

    @property
    def model_name(self) -> str:
        """
        Get the model name.
        Returns
        -------
        str
            Model name.
        """
        return self._model_name

    def forward(self, data: Data, mode: str = "train") -> Tuple[Tensor, Any]:
        """
        GAT Forward pass.
        Parameters
        ----------
        data : Data
            Input dataset.
        mode : bool
            Which subset to forward pass through.
        Returns
        -------
        Tuple[Tensor, Any]
        """
        x, edge_index = data.x, data.edge_index
        x_out = self.model(x, edge_index)

        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        # loss = F.nll_loss(x_out[mask], data.y[mask])
        loss = self.loss_fn(x_out[mask], data.y[mask])
        pred = x_out[mask].argmax(-1)
        accuracy = (pred == data.y[mask]).sum() / pred.shape[0]

        return loss, accuracy

    def training_step(self, batch: Data, batch_index: int) -> Tensor:
        """
        Model training step.
        Parameters
        ----------
        batch : Data
            Data batch.
        batch_index : int
            Batch index.
        Returns
        -------
        Tensor
        """
        loss, accuracy = self.forward(batch, mode="train")

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch: Data, batch_index: int) -> Tensor:
        """
        Model validation step.
        Parameters
        ----------
        batch : Data
            Data batch.
        batch_index : int
            Batch index.
        Returns
        -------
        Tensor
        """
        loss, accuracy = self.forward(batch, mode="val")

        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)

        return loss

    def test_step(self, batch: Data, batch_index: int) -> Tensor:
        """
        Model test step.
        Parameters
        ----------
        batch : Data
            Data batch.
        batch_index : int
            Batch index.
        Returns
        -------
        Tensor
        """
        loss, accuracy = self.forward(batch, mode="test")

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure model optimizer.
        Returns
        -------
        torch.optim.Optimizer
            Model optimizer.
        """
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
