from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch_geometric.data import Data

from src.data_modules import NodeDataModule
from src.models.node_classification.base import BaseNodeClassifier
from src.utils import get_jacobian, load_config

global_config = load_config("global_config.yaml")
training_config = load_config("training_config.yaml")


def plot_dirichlet_energies(
    _data: NodeDataModule,
    _model: BaseNodeClassifier,
    model_dirichlet_energies: List[float],
) -> None:
    """
    Plot the Dirichlet energy against the layer ID of a given trained model.
    Parameters
    ----------
    _data: NodeDataModule
        Dataset used by neural network.
    _model : BaseNodeClassifier
        Model whose energies to plot.
    model_dirichlet_energies : List[float]
        List of Dirichlet energies gathered during training.

    Returns
    -------
    None
    """
    plt.plot(model_dirichlet_energies, color="black")
    plt.title(
        f"{_data.dataset_name}: {_model.model_name}-{_model.n_hidden} Hidden"
        f"- Dirichlet Energy",
        fontsize=14,
    )
    plt.xlabel("Layer ID")
    plt.ylabel("Dirichlet Energy")

    save_dir = Path(
        training_config["save_plots_dir"],
        f"{_data.dataset_name}_results",
        "energy",
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        Path(save_dir, f"{_model.model_name}_{_model.n_hidden}h.png"),
        dpi=global_config["fig_dpi"],
    )


def plot_influences(
    _model: BaseNodeClassifier, _data_module: NodeDataModule, _data: Data
) -> None:
    """
    Plot the influence of k-hop neighbors on a node after training the model.
    Parameters
    ----------
    _model : BaseNodeClassifier
        Trained model.
    _data_module: NodeDataModule
        Data module of dataset used by neural network.
    _data : Data
        Dataset containing node and neighbors to plot.

    Returns
    -------
    None
    """
    n_nodes_influence = training_config["n_nodes_influence"]
    i, r = (
        np.random.choice(_data.x.shape[0], size=n_nodes_influence),
        10,
    )

    fig, ax = plt.subplots(
        1, n_nodes_influence, figsize=(5 * n_nodes_influence, 4)
    )

    for j, val in enumerate(i):
        influences = []
        for k in range(1, r + 1):
            influence_dist = get_jacobian(_model, _data, val, k)
            if influence_dist["influence"].isnull().values.any():
                continue

            influences.append(influence_dist)

        influences_df = pd.concat(influences)

        sns.violinplot(
            data=influences_df.reset_index(drop=True),
            x="r",
            y="influence",
            color="black",
            ax=ax[j],
        )
        ax[j].set_title(f"Jacobian at r = {r}, Node = {val}", fontsize=12)

    plt.suptitle(
        f"{_model.model_name}-{_model.n_hidden} Hidden - Influences",
        fontsize=14,
    )

    save_dir = Path(
        training_config["save_plots_dir"],
        f"{_data_module.dataset_name}_results",
        "neighbor_influences",
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        Path(save_dir, f"{_model.model_name}_{_model.n_hidden}h.png"),
        dpi=global_config["fig_dpi"],
    )
