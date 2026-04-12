from __future__ import annotations

import torch
import yaml
from pathlib import Path
from typing import Any

def get_device() -> torch.device:
    """Gets the available computation device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Loads configuration from a YAML file.

    Args:
        config_path (str | Path): Path to the YAML configuration file.

    Returns:
        dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open("r") as f:
        return yaml.safe_load(f)
