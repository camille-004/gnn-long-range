from typing import Any, Callable, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data


class BaseGraphClassifier(pl.LightningModule):
    """Base graph classifier."""

    def __init__(
        self, n_hidden: int, activation: nn.Module() = None, **kwargs
    ) -> None:
        super().__init__()
        pl.utilities.seed.seed_everything(1)

        assert n_hidden >= 0, "Number of hidden layers must be non-negative."
        self.n_hidden = n_hidden
        self.activation = activation

        self._model_name = "base_graph_clf"

        self.num_features: int = 0
        self.num_classes: int = 0

        self.loss_fn: Callable[[Tensor, Tensor], Tensor] = F.cross_entropy
        self.lr = 0.01
        self.weight_decay = 5e-4

        print("\n=====\nMODEL\n=====\n")

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

    def forward(self, x: Any, edge_index: Any, batch_index: Any) -> Tensor:
        x_out = self.model(x, edge_index, batch_index)
        return x_out

    def step_util(self, batch: Data) -> Tuple[Tensor, Any]:
        """
        Model step util. To be used for training, validation, and testing.
        Parameters
        ----------
        batch : Data
            Data batch.
        Returns
        -------
        Tensor
        """
        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch

        x_out = self.forward(x, edge_index, batch_index)
        loss = self.loss_fn(x_out, batch.y)

        # Metrics
        pred = x_out.argmax(-1)
        accuracy = (pred == batch.y).sum() / pred.shape[0]

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
        loss, accuracy = self.step_util(batch)

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
        loss, accuracy = self.step_util(batch)

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
        loss, accuracy = self.step_util(batch)

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
