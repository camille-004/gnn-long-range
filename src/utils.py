from pathlib import Path
from typing import Any, Dict, List, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.convert import to_networkx

from src.models.graph_classification.base import BaseGraphClassifier
from src.models.node_classification.base import BaseNodeClassifier

CONFIG_DIR = Path(Path(__file__).parent.parent, "config")


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load the contents of the configuration file.
    Parameters
    ----------
    config_name : str
        Path to configuration file.
    Returns
    -------
    Dict[str, Any]
        Contents of the JSON configuration file.
    """
    config_path = Path(CONFIG_DIR, config_name)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def get_graph_laplacian(
    edge_index: Tensor, num_nodes: int, normalization: str = "sym"
) -> torch.sparse.FloatTensor:
    edge_index, edge_weight = get_laplacian(
        edge_index, normalization=normalization
    )
    return torch.sparse.FloatTensor(
        edge_index, edge_weight, torch.Size([num_nodes, num_nodes])
    )


def dirichlet_energy(x: Tensor, laplacian: Tensor) -> np.ndarray:
    """
    Calculate the Dirichlet energy at a NN layer.

    Parameters
    ----------
    x : Tensor
        Node features.
    laplacian : Tensor
        Graph Laplacian

    Returns
    -------
    np.ndarray
        Dirichlet energy value.
    """
    _L = np.array(laplacian.cpu().to_dense())
    assert x.shape[0] == _L.shape[0] == _L.shape[1]
    x = x.clone().detach().cpu()  # Don't calculate gradient
    _E = np.dot(np.dot(x.T, _L), x)
    _E = np.diag(_E)
    return np.sum(_E)


def k_hop_nb(data: Data, node: int, r: int) -> List[int]:
    """
    Return the list of nodes that are of distance r from a give node.

    Parameters
    ----------
    data : Data
        Input graph.
    node : int
        Node whose r-th-order neighborhood to return.
    r : int
        Order of neighborhood.

    Returns
    -------
    List[int]
        List of r-hop neighborhood members.
    """
    _G = to_networkx(data)
    path_lengths = nx.single_source_dijkstra_path_length(_G, node)
    return [node for node, length in path_lengths.items() if length == r]


def get_jacobian(
    _model: Union[BaseGraphClassifier, BaseNodeClassifier],
    _data: Data,
    node: int,
    r: int,
) -> pd.DataFrame:
    """
    Get the Jacobian of the embeddings of neighbors at a distance r from node
    i w.r.t. the features at node i. This will assess the effect of
    over-squashing.

    Parameters
    ----------
    _model : Union[BaseGraphClassifier, BaseNodeClassifier]
        Model from which to obtain embeddings.
    _data : Data
        Input evaluation set.
    node : int
        Index of the fixed node.
    r : int
        Order of neighborhood.

    Returns
    -------
    pd.DataFrame
        Distribution of influences of neighbors at a distance r from node i.
    """
    if not _data.x.requires_grad:
        _data.x.requires_grad = True

    neighbor_nodes_idx = k_hop_nb(_data, node, r)
    # print(f"Number of {r}-hop neighbors: {len(neighbor_nodes_idx)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _data = _data.to(device)
    _model = _model.to(device)
    embeddings = _model(_data.x, _data.edge_index)[1]
    sum_of_grads = torch.autograd.grad(
        embeddings[node],
        _data.x,
        torch.ones_like(embeddings[node]),
        retain_graph=True,
    )[0][neighbor_nodes_idx]
    abs_grad = sum_of_grads.absolute()
    sum_of_jacobian = abs_grad.sum(axis=1)
    influence_y_on_x = sum_of_jacobian / sum_of_jacobian.sum(dim=0)
    influence_y_on_x = influence_y_on_x.cpu().numpy()
    return pd.DataFrame(data={"influence": influence_y_on_x, "r": r})
