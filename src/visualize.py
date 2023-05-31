from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch_geometric.data import Data

from .data.data_module import NodeDataModule
from .models.base import BaseNodeClassifier
from .models.utils import get_jacobian
from .utils import load_config

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
    augmentation_thres = _data.add_edges_thres
    plt.figure()
    plt.plot(model_dirichlet_energies, color="black")
    plt.suptitle(
        f"{_data.dataset_name}: {_model.model_name}-{_model.n_hidden+1} Hidden"
        f" - log(Dirichlet Energy)",
        fontsize=10,
    )
    plt.title(f"add_edges_thres = {augmentation_thres}", fontsize=10)
    plt.xlabel("Layer ID")
    plt.ylabel("log(Dirichlet Energy)")

    save_dir = Path(
        training_config["save_plots_dir"],
        f"{_data.dataset_name}_results",
        "energy",
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    img_name = f"{_model.model_name}_{_model.n_hidden}h"
    if hasattr(_model, "num_heads"):
        img_name += f"_{_model.num_heads}_head"

    plt.savefig(
        Path(save_dir, f"{img_name}.png"),
        dpi=global_config["fig_dpi"],
    )


def plot_rayleigh_quotients(
    _data: NodeDataModule,
    _model: BaseNodeClassifier,
    model_rayleigh_quotients: List[float],
) -> None:
    """
    Plot the Rayleigh quotient against the layer ID of a given trained model.
    Parameters
    ----------
    _data: NodeDataModule
        Dataset used by neural network.
    _model : BaseNodeClassifier
        Model whose energies to plot.
    model_rayleigh_quotients : List[float]
        List of Rayleigh quotients gathered during training.

    Returns
    -------
    None
    """
    augmentation_thres = _data.add_edges_thres
    plt.figure()
    plt.plot(model_rayleigh_quotients, color="blue")
    plt.suptitle(
        f"{_data.dataset_name}: {_model.model_name}-{_model.n_hidden} Hidden"
        f" - Rayleigh Quotient",
        fontsize=10,
    )
    plt.title(f"add_edges_thres = {augmentation_thres}", fontsize=10)
    plt.xlabel("Layer ID")
    plt.ylabel("Rayleigh Quotient")

    save_dir = Path(
        training_config["save_plots_dir"],
        f"{_data.dataset_name}_results",
        "rayleigh",
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    img_name = f"{_model.model_name}_{_model.n_hidden}h_{augmentation_thres}"
    if hasattr(_model, "num_heads"):
        img_name += f"_{_model.num_heads}_head"
    if hasattr(_model, "r"):
        img_name += f"_{_model.r}_r"

    plt.savefig(
        Path(save_dir, f"{img_name}.png"),
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
    augmentation_thres = _data_module.add_edges_thres

    n_nodes_influence = training_config["n_nodes_influence"]
    i, r = (
        np.random.choice(_data.x.shape[0], size=n_nodes_influence),
        10,
    )

    fig, ax = plt.subplots(
        1, n_nodes_influence, figsize=(5 * n_nodes_influence, 4), sharex=True
    )

    if n_nodes_influence > 1:
        influences_ = []
        for j, val in enumerate(i):
            influences = []
            influencess = []
            for k in range(1, r + 1):
                influence_dist, influence_sum = get_jacobian(_model, _data, val, k)
                if influence_dist["influence"].isnull().values.any():
                    continue

                influences.append(influence_dist)
                influencess.append(influence_sum)
            influences_.append(influencess)

            if len(influences) == 0:
                continue

            influences_df = pd.concat(influences)

            try:
                sns.violinplot(
                    data=influences_df.reset_index(drop=True),
                    x="r",
                    y="influence",
                    color="blue",
                    ax=ax[j],
                )
            except ValueError:
                print("Isolated node. Could not plot influences.")
                return

            ax[j].set_title(f"Jacobian at r = {r}, Node = {val}", fontsize=12)

        plt.suptitle(
            f"{_model.model_name}-{_model.n_hidden+1} Hidden - Influences",
            fontsize=10,
        )
        # plt.title(f"add_edges_thres = {augmentation_thres}", fontsize=10)
    else:
        influences = []
        influences_ = []
        for k in range(1, r + 1):
            influence, influence_sum = get_jacobian(_model, _data, i[0], k)
            influences.append(influence)
            influences_.append(influence_sum)
        influences_df = pd.concat(influences)

        try:
            # sns.violinplot(
            #     data=influences_df.reset_index(drop=True),
            #     x="r",
            #     y="influence",
            #     color="blue",
            # )
            sns.boxplot(
                data=influences_df.reset_index(drop=True),
                x="r",
                y="influence",
                color="blue",
            )
        except ValueError:
            print("Isolated node. Could not plot influences.")
            return

        plt.title(
            f"{_model.model_name}-{_model.n_hidden+1}\nJacobian at r = {r}, "
            f"Node = {i}",
            fontsize=10,
        )

    print(f"Inlfuences: {influences_}")
    save_dir = Path(
        training_config["save_plots_dir"],
        f"{_data_module.dataset_name}_results",
        "neighbor_influences",
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    img_name = f"{_model.model_name}_{_model.n_hidden}h"
    if hasattr(_model, "num_heads"):
        img_name += f"_{_model.num_heads}_head"
    if hasattr(_model, "r"):
        img_name += f"_{_model.r}_r"
    img_name += f"_{n_nodes_influence}_nodes"

    plt.savefig(
        Path(save_dir, f"{img_name}.png"),
        dpi=global_config["fig_dpi"],
    )
