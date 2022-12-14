from pathlib import Path
from typing import Any, Dict

import yaml

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
