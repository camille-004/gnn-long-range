from typing import Any, Callable, List, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from .sognn_layer import SOGNNConv

from .utils import dirichlet_energy, get_graph_laplacian, rayleigh_quotient

Mode = Literal["train", "val", "test"]


class BaseNodeClassifier(pl.LightningModule):
    """Base node classifier."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int,
        n_hidden: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        activation: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        pl.utilities.seed.seed_everything(1)

        self._model_name: str = "base_node_clf"
        self.criterion: Callable[
            [Tensor, Tensor], Tensor
        ] = nn.CrossEntropyLoss()
        self.energies: Optional[List[float]] = None
        self.rayleigh: Optional[List[float]] = None

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation

        assert n_hidden >= 0, "Number of hidden layers must be non-negative."
        self.n_hidden = n_hidden

        self.convs = nn.ModuleList()

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

    def forward(self, x: Any, edge_index: Any) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Any
            Node features.
        edge_index : Any
            Adjacency matrix.

        Returns
        -------
        Tensor
            Output of the forward pass.
        """
        _L = get_graph_laplacian(edge_index, x.size(0))
        self.energies = []
        self.rayleigh = []

        if self._model_name == "node_SOGNN":
            SOGNNConv.set_distant_adjacency_matrix(edge_index=edge_index)

        for i in range(self.n_hidden + 2):

            x = self.convs[i](x, edge_index)
            energy = dirichlet_energy(x, _L)
            rayleigh = rayleigh_quotient(x, _L)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            self.energies.append(energy)
            self.rayleigh.append(rayleigh)

        return F.log_softmax(x, dim=1), x

    def step_util(self, batch: Data, mode: Mode) -> Tuple[Tensor, Any]:
        """
        Model step util. To be used for training, validation, and testing.

        Parameters
        ----------
        batch : Data
            Data batch.
        mode : Mode
            Mask of input data.

        Returns
        -------
        Tensor
        """
        x, edge_index = batch.x, batch.edge_index
        x_out = self.forward(x, edge_index)

        if isinstance(x_out, tuple):
            x_out = x_out[0]

        if mode == "train":
            mask = batch.train_mask
        elif mode == "val":
            mask = batch.val_mask
        elif mode == "test":
            mask = batch.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        loss = self.criterion(x_out[mask], batch.y[mask])

        # Metrics
        pred = x_out[mask].argmax(-1)
        accuracy = (pred == batch.y[mask]).sum() / pred.shape[0]

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
        loss, accuracy = self.step_util(batch, "train")

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
        loss, accuracy = self.step_util(batch, "val")

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
        loss, accuracy = self.step_util(batch, "test")

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

    def get_energies(self) -> List[float]:
        """
        Get the list of Dirichlet energies at each layer after training the
        model.

        Returns
        -------
        List[float]
            List of Dirichlet energies.
        """
        if self.energies is None:
            raise RuntimeError(
                f"Train {self.model_name} the to get Dirichlet energies."
            )

        return self.energies

    def get_rayleigh(self) -> List[float]:
        """
        Get the list of Rayleigh quotients at each layer after training the
        model.

        Returns
        -------
        List[float]
            List of Rayleigh quotients.
        """
        if self.rayleigh is None:
            raise RuntimeError(
                f"Train {self.model_name} the to get Rayleigh quotients."
            )

        return self.rayleigh
