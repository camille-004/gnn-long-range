from typing import Optional

import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

from ..utils import load_config
from .add_edges import AddRandomEdges

config = load_config("data_config.yaml")
DATA_DIR = config["data_dir"]

graph_data_config = config["graph"]
node_data_config = config["node"]


class NodeDataModule(pl.LightningDataModule):
    """Data module for node classification datasets, particularly
    Planetoid datasets."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.dataset = None

        self._dataset_name = (
            kwargs["dataset_name"]
            if "dataset_name" in kwargs.keys()
            else node_data_config["node_data_name_default"]
        )
        self.add_edges_thres = (
            kwargs["add_edges_thres"]
            if "add_edges_thres" in kwargs.keys()
            else node_data_config["add_edges_thres"]
        )
        self.norm_features = (
            kwargs["norm_features"]
            if "norm_features" in kwargs.keys()
            else config["norm_features_default"]
        )
        self.batch_size: int = (
            kwargs["batch_size"]
            if "batch_size" in kwargs.keys()
            else config["batch_size_default"]
        )
        self.num_cpus: int = (
            kwargs["num_cpus"]
            if "num_cpus" in kwargs.keys()
            else config["num_cpus_default"]
        )

        self.transform = [AddRandomEdges(self.add_edges_thres)]

        if self.norm_features:
            self.transform.append(T.NormalizeFeatures())

        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Define the dataset.
        Parameters
        ----------
        stage : Optional[str]
            Dataset stage, not used.
        Returns
        -------
        None
        """
        self.dataset = Planetoid(
            root=f"{DATA_DIR}/{self.dataset_name}",
            name=self.dataset_name,
            transform=T.Compose(self.transform),
        )

    @property
    def dataset_name(self) -> str:
        """
        Return the dataset name.
        Returns
        -------
        str
            Dataset name.
        """
        return self._dataset_name

    @property
    def num_features(self) -> int:
        """
        Return the dimension of node features.
        Returns
        -------
        int
            The dimension of node features.
        """
        return self.dataset.num_features

    @property
    def num_classes(self) -> int:
        """
        Return the number of classes.
        Returns
        -------
        int
            The number of classes.
        """
        return self.dataset.num_classes

    def train_dataloader(self) -> DataLoader:
        """
        Define train DataLoader.
        Returns
        -------
        DataLoader
            Train DataLoader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Define validation DataLoader.
        Returns
        -------
        DataLoader
            Validation DataLoader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Define test DataLoader.
        Returns
        -------
        DataLoader
            Test DataLoader.
        """
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
            shuffle=False,
        )
