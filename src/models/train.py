import warnings
from typing import Dict, Tuple, Type, Union

import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.data_modules import GraphDataModule, NodeDataModule
from src.models.graph_classification.base import BaseGraphClassifier
from src.models.graph_classification.gat import GraphLevelGAT
from src.models.graph_classification.gcn import GraphLevelGCN

# isort: off
from src.models.graph_classification.gin import (
    GraphLevelGIN,
    GraphLeveLGINWithJK,
)

# isort: on
from src.models.node_classification.base import BaseNodeClassifier
from src.models.node_classification.gat import NodeLevelGAT
from src.models.node_classification.gcn import NodeLevelGCN
from src.models.node_classification.gin import NodeLevelGIN
from src.utils import load_config
from src.visualize import plot_dirichlet_energies, plot_influences

data_config = load_config("data_config.yaml")
training_config = load_config("training_config.yaml")
global_config = load_config("global_config.yaml")

sns.set_style(global_config["sns_style"])


def get_model(
    task: str, model_name: str
) -> Type[Union[BaseGraphClassifier, BaseNodeClassifier]]:
    """
    Return the type of model to use from the model name.

    Parameters
    ----------
    task : str
        Either "node" or "graph" for node or graph classification,
        respectively.
    model_name : str
        Name of model to train.

    Returns
    -------
    Union[BaseGraphClassifier, BaseNodeClassifier]
        Class of the model to train.
    """
    model_map = {
        "graph": {
            "gat": GraphLevelGAT,
            "gcn": GraphLevelGCN,
            "gin": GraphLevelGIN,
            "gin_jk": GraphLeveLGINWithJK,
        },
        "node": {
            "gat": NodeLevelGAT,
            "gcn": NodeLevelGCN,
            "gin": NodeLevelGIN,
        },
    }

    if task == "graph":
        assert model_name in model_map["graph"].keys()
    else:
        assert model_name in model_map["node"].keys()

    return model_map[task][model_name]


def get_activation(activation: str = "relu") -> nn.Module:
    """
    Get the instance of an activation function from the function name.

    Parameters
    ----------
    activation : str
        Name of the activation function.

    Returns
    -------
    nn.Module
        Instance of the activation function

    """
    if activation == "relu":
        warnings.warn(
            f"Non-differentiable function: {activation}. Might not be able to "
            f"get accurate influences."
        )
    activation_map = {"relu": nn.ReLU(), "elu": nn.ELU(), "tanh": nn.Tanh()}

    assert activation in activation_map.keys(), "Unknown activation function."
    return activation_map[activation]


def prepare_training(
    task: str,
    _model: str,
    n_hidden: int,
    activation: str = None,
    dataset_name: str = data_config["node"]["node_data_name_default"],
    **kwargs,
) -> Tuple[
    Union[GraphDataModule, NodeDataModule],
    Union[BaseGraphClassifier, BaseNodeClassifier],
]:
    """
    Return DataModule and specified model for the input task.
    Parameters
    ----------
    task : str
        Either "node" or "graph" for node or graph classification,
        respectively.
    _model : str
        Name of model to train.
    n_hidden : int
        Number of hidden layers in the neural network.
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
    assert task in ["graph", "node"], "Unknown task."

    if task == "graph":
        _data_module = GraphDataModule(dataset_name=dataset_name)
    else:
        _data_module = NodeDataModule(dataset_name=dataset_name)

    if activation is not None:
        activation_fn = get_activation(activation)
    else:
        activation_fn = None

    model_type = get_model(task, _model)
    model_instance = model_type(
        n_hidden=n_hidden,
        activation=activation_fn,
        num_features=_data_module.num_features,
        num_classes=_data_module.num_classes,
        **kwargs,
    )
    return _data_module, model_instance


def train_module(
    _data_module: Union[GraphDataModule, NodeDataModule],
    _model: Union[BaseGraphClassifier, BaseNodeClassifier],
    use_early_stopping: bool = True,
    early_stopping_patience: int = training_config[
        "early_stopping_patience_default"
    ],
    early_stopping_min_delta: float = training_config[
        "early_stopping_min_delta_default"
    ],
    max_epochs: int = training_config["max_epochs_default"],
    plot_energies: bool = False,
    plot_influence: bool = False,
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

    project_name = f"{_data_module.dataset_name}_{_model.model_name}"
    wandb_logger = WandbLogger(project=project_name, log_model="all")

    callbacks = []

    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )
        callbacks.append(early_stopping)

    model_checkpoint = ModelCheckpoint(monitor="val_accuracy", mode="max")
    callbacks.append(model_checkpoint)
    device = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        max_epochs=max_epochs,
        accelerator=device,
    )

    trainer.fit(model=_model, datamodule=_data_module)
    val_data = next(iter(_data_module.val_dataloader()))

    if plot_energies:
        model_dirichlet_energies = _model.get_energies()
        plot_dirichlet_energies(_data_module, _model, model_dirichlet_energies)

    if plot_influence:
        plot_influences(_model, _data_module, val_data)

    val_results = trainer.validate(_model, datamodule=_data_module)
    test_results = trainer.test(_model, datamodule=_data_module)
    val_results.extend(test_results)
    val_results[0].update(val_results[1])
    model_results = val_results[0]

    wandb.finish()
    return {_model.model_name: model_results}
