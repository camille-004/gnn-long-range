from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch_geometric.data import Data

from src.models.graph_classification.base import BaseGraphClassifier
from src.models.node_classification.base import BaseNodeClassifier
from src.utils import get_jacobian, load_config

training_config = load_config("training_config.yaml")


def plot_dirichlet_energies(
    _model: Union[BaseGraphClassifier, BaseNodeClassifier],
    model_dirichlet_energies: List[float],
) -> None:
    """
    Plot the Dirichlet energy against the layer ID of a given trained model.
    Parameters
    ----------
    _model : Union[BaseGraphClassifier, BaseNodeClassifier]
        Model whose energies to plot.
    model_dirichlet_energies : List[float]
        List of Dirichlet energies gathered during training.

    Returns
    -------
    None
    """
    plt.plot(model_dirichlet_energies, color="black")
    plt.title(
        f"{_model.model_name}-{_model.n_hidden} Hidden - Dirichlet Energy",
        fontsize=14,
    )
    plt.xlabel("Layer ID")
    plt.ylabel("Dirichlet Energy")
    plt.show()


def plot_influences(
    _model: Union[BaseGraphClassifier, BaseNodeClassifier], _data: Data
) -> None:
    """
    Plot the influence of k-hop neighbors on a node after training the model.
    Parameters
    ----------
    _model : Union[BaseGraphClassifier, BaseNodeClassifier]
        Trained model.
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
        influences = pd.DataFrame()
        for k in range(1, r + 1):
            influence_dist = get_jacobian(_model, _data, val, k)
            if influence_dist["influence"].isnull().values.any():
                continue

            influences = influences.append(influence_dist)

        sns.violinplot(
            data=influences.reset_index(drop=True),
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
    plt.show()
