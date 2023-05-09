from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_networkx
import pickle


def write_edge_index(edge_index: Tensor, path: str):
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(edge_index, f)
    with open(f'{path}.data', 'w') as b:
        for i in range(edge_index.shape[1]):
            print(f'{edge_index[0][i].item()}', file=b, end=' '*(10-len(str(edge_index[0][i].item()))))
            print(f'{edge_index[1][i].item()}', file=b)


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
    return np.log(np.sum(_E))


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
    _model: pl.LightningModule,
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
    )[0]
    abs_grad = sum_of_grads[neighbor_nodes_idx].absolute()
    sum_of_jacobian = abs_grad.sum(axis=1)
    if sum_of_jacobian.sum(dim=0).item():
        influence_sum = sum_of_jacobian.sum(dim=0).item()
        influence_y_on_x = sum_of_jacobian / sum_of_jacobian.sum(dim=0)
        influence_y_on_x = influence_y_on_x.cpu().numpy()
    else:
        influence_y_on_x = torch.zeros_like(sum_of_jacobian).cpu().numpy()
        influence_sum = 0
    return pd.DataFrame(data={"influence": sum_of_jacobian.cpu().numpy(), "r": r}), influence_sum


def rayleigh_quotient(x: Tensor, laplacian: Tensor) -> np.ndarray:
    """
    Calculate the Rayleigh quotient at a NN layer.

    NOTE: nan in some dimensions. May not be accurate!

    Parameters
    ----------
    x : Tensor
        Node features.
    laplacian : Tensor
        Graph Laplacian

    Returns
    -------
    np.ndarray
        Rayleigh quotient value.
    """
    d = x.size(1)
    _L = np.array(laplacian.cpu().to_dense())
    x = x.clone().detach().cpu().numpy()  # Don't calculate gradient
    _R = []

    for i in range(d):
        energy = np.dot(np.dot(x[:, i].T, _L), x[:, i])  # Scalar
        norm = np.dot(x[:, i].T, x[:, i])
        _R.append(energy / norm)

    return np.nanmean(_R)
