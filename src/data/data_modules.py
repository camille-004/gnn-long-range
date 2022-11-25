from typing import Optional

import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

from src.utils import load_config

config = load_config("data_config.json")
DATA_DIR = config["data_dir"]

graph_data_config = config["graph"]
node_data_config = config["node"]


class NodeDataModule(pl.LightningDataModule):
    """Data module for node classification datasets, particularly
    Planetoid datasets."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.dataset = None

        self.dataset_name = (
            kwargs["dataset_name"]
            if "dataset_name" in kwargs.keys()
            else node_data_config["node_data_name_default"]
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

        self.transform = []

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


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.dataset_name = (
            kwargs["dataset_name"]
            if "dataset_name" in kwargs.keys()
            else graph_data_config["graph_data_name_default"]
        )
        self.norm_features = (
            kwargs["norm_features"]
            if "norm_features" in kwargs.keys()
            else config["norm_features_default"]
        )
        self.batch_size = (
            kwargs["batch_size"]
            if "batch_size" in kwargs.keys()
            else config["batch_size_default"]
        )
        self.num_cpus = (
            kwargs["num_cpus"]
            if "num_cpus" in kwargs.keys()
            else config["num_cpus_default"]
        )

        self.transform = []
        self.root = f"{DATA_DIR}/{self.dataset_name}"

        if self.norm_features:
            self.transform.append(T.NormalizeFeatures())

        if self.dataset_name in ["IMDB-BINARY", "PROTEINS"]:
            self.transform.insert(
                0, T.OneHotDegree(graph_data_config["imdb_features"])
            )

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
        dataset = TUDataset(
            root=self.root,
            name=self.dataset_name,
            transform=T.Compose(self.transform),
        )

        dataset = dataset.shuffle()

        # 80/10/10 split
        num_samples = len(dataset)
        num_val = num_samples // 10

        self.train_dataset = dataset[2 * num_val :]
        self.val_dataset = dataset[:num_val]
        self.test_dataset = dataset[num_val : 2 * num_val]

    @property
    def num_features(self) -> int:
        """
        Return the dimension of node features.
        Returns
        -------
        int
            The dimension of node features.
        """
        return self.train_dataset.num_features

    @property
    def num_classes(self) -> int:
        """
        Return the number of classes.
        Returns
        -------
        int
            The number of classes.
        """
        return self.train_dataset.num_classes

    def train_dataloader(self) -> DataLoader:
        """
        Define train DataLoader.
        Returns
        -------
        DataLoader
            Train DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
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
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
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
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
        )
