import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform("add_random_edges")
class AddRandomEdges(BaseTransform):
    """Custom PyG Transform to add random edges."""

    def __init__(self, thres: float = 0.0):
        self.thres = thres

    def __call__(self, data: Data) -> Data:
        """
        Randomly add |E| * thres edges to a PyG graph.

        Parameters
        ----------
        data : Data
            Input PyG graph.

        Returns
        -------
        Data
            PyG graph with new edges.
        """
        edges = data.edge_index.T.numpy()
        new_edges = []

        while len(new_edges) <= int(len(edges) * self.thres):
            n = data.x.shape[0]
            a, b = np.random.choice(n, size=2, replace=False)
            new_edges.append([a, b])

        data.edge_index = torch.LongTensor(
            np.concatenate([edges, new_edges]).T
        )
        return data
