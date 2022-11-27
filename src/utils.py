from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch import Tensor

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


def dirichlet_energy(x: Tensor, laplacian: Tensor) -> float:
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
    float
        Dirichlet energy value.
    """
    return torch.trace(torch.matmul(torch.matmul(x.t(), laplacian), x)).item()