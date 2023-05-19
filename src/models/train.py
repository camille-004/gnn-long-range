import csv
from pathlib import Path
from typing import Dict, Tuple, Type

import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

from ..data.data_module import NodeDataModule
from ..types.enums import Activation, Model
from ..utils import load_config
from ..visualize import (plot_dirichlet_energies, plot_influences,
                         plot_rayleigh_quotients)
from .base import BaseNodeClassifier
from .gat import NodeLevelGAT
from .gcn import NodeLevelGCN
from .gin import NodeLevelGIN
from .sognn import NodeLevelSOGNN
from .sognn_layer import SOGNNConv

data_config = load_config("data_config.yaml")
model_config = load_config("model_config.yaml")
training_config = load_config("training_config.yaml")
global_config = load_config("global_config.yaml")

sns.set_style(global_config["sns_style"])


ACTIVATION_MAP: Dict[Activation, nn.Module] = {
    Activation.ELU: nn.ELU(),
    Activation.RELU: nn.ReLU(),
    Activation.TANH: nn.Tanh(),
}

MODEL_MAP: Dict[Model, Type[BaseNodeClassifier]] = {
    Model.GAT: NodeLevelGAT,
    Model.GCN: NodeLevelGCN,
    Model.GIN: NodeLevelGIN,
    Model.SOGNN: NodeLevelSOGNN
}


def prepare_training(
    _model: str,
    n_hidden: int,
    add_edges_thres: float,
    activation: str = None,
    dataset_name: str = data_config["node"]["node_data_name_default"],
    **kwargs,
) -> Tuple[NodeDataModule, BaseNodeClassifier,]:
    """
    Return DataModule and specified model for the input task.
    Parameters
    ----------
    task : Task
        Either "node" or "graph" for node or graph classification,
        respectively.
    _model : str
        Name of model to train.
    n_hidden : int
        Number of hidden layers in the neural network.
    add_edges_thres : float
        Threshold (as a percentage of original edge cardinality) of edges to
        randomly add to the input graph.
    activation : str
        Activation function to use in the neural network.
    dataset_name : str
        Name of dataset name in DataModule.

    Returns
    -------
    Tuple[
        Union[datasets.GraphDataModule, datasets.NodeDataModule],
        Union[BaseGraphClassifier, BaseNodeClassifier]
    ]
        DataModule and model instance for training.
    """
    if activation is not None:
        activation = ACTIVATION_MAP[activation]


    _data_module = NodeDataModule(
        dataset_name=dataset_name, add_edges_thres=add_edges_thres
    )

    model_instance = MODEL_MAP[_model](
        _data_module.num_features,
        _data_module.num_classes,
        model_config["hidden_channels_default"],
        n_hidden,
        model_config["dropout"],
        model_config["lr"],
        model_config["weight_decay"],
        activation=activation,
        **kwargs,
    )
    return _data_module, model_instance


def train_module(
    _data_module: NodeDataModule,
    _model: BaseNodeClassifier,
    use_early_stopping: bool = True,
    early_stopping_patience: int = training_config[
        "early_stopping_patience_default"
    ],
    early_stopping_min_delta: float = training_config[
        "early_stopping_min_delta_default"
    ],
    max_epochs: int = training_config["max_epochs_default"],
    plot_energies: bool = False,
    plot_rayleigh: bool = False,
    plot_influence: bool = False,
    r:int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Set up WandB logger and PTL Trainer, train input model on input dataset,
    and return validation and test results.
    Parameters
    ----------
    _data_module : pl.LightningDataModule
        Graph neural network dataset.
    _model : pl.LightningModule
        Graph neural network model.
    use_early_stopping : bool
        Whether to use early stopping.
    early_stopping_patience : int
        Patience for early stopping.
    early_stopping_min_delta : float
        Minimum delta for early stopping.
    max_epochs : int
        Maximum number of epochs for training.
    plot_energies : bool
        Whether to plot Dirichlet energy after training.
    plot_rayleigh : bool
        Whether to plot Rayleigh quotients after training.
    plot_influence : bool
        Whether to plot the influence of k-hop neighbors on a node x.
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary of validation and testing results.
    """
    _num_features = _data_module.num_features
    _num_classes = _data_module.num_classes
    print("==================\nDATASET STATISTICS\n==================\n")
    print(f"Number of features: {_num_features}")
    print(f"Number of classes: {_num_classes}\n")

    print("========\nTRAINING\n========\n")
    print(_model)

    project_name = f"{_data_module.dataset_name}_{_model.model_name}"
    wandb_logger = WandbLogger(project=project_name, log_model="all")

    callbacks = []

    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            verbose=True,
        )
        callbacks.append(early_stopping)

    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
    callbacks.append(model_checkpoint)
    device = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        max_epochs=max_epochs,
        accelerator=device,
    )
    print("==============\nTRAINING START\n==============\n")
    trainer.fit(model=_model, datamodule=_data_module)
    val_data = next(iter(_data_module.val_dataloader()))

    if plot_energies:
        print("Plotting energies...")
        model_dirichlet_energies = _model.get_energies()
        plot_dirichlet_energies(_data_module, _model, model_dirichlet_energies)

    if plot_rayleigh:
        print("Plotting Rayleigh quotients...")
        model_rayleigh_quotients = _model.get_rayleigh()
        plot_rayleigh_quotients(_data_module, _model, model_rayleigh_quotients)

    if plot_influence:
        print("Plotting influences...")
        plot_influences(_model, _data_module, val_data)

    val_results = trainer.validate(_model, datamodule=_data_module)
    test_results = trainer.test(_model, datamodule=_data_module)
    val_results.extend(test_results)
    val_results[0].update(val_results[1])
    model_results = val_results[0]

    results_path = Path(
        global_config["logs_dir"], global_config["results_name"]
    )

    with open(results_path, "a") as f:
        writer = csv.writer(f)
        row = [
            _model.model_name,
            _data_module.dataset_name,
            _model.n_hidden,
            # _data_module.add_edges_thres,
            np.round(model_results["val_loss"], 4),
            np.round(model_results["val_accuracy"], 4),
            np.round(model_results["test_loss"], 4),
            np.round(model_results["test_accuracy"], 4),
            type(_model.activation).__name__,
            r,
        ]

        if hasattr(_model, "num_heads"):
            row[-1] = _model.num_heads

        writer.writerow(row)

    wandb.finish()
    return {_model.model_name: model_results}
