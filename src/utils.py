from __future__ import annotations

import os
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

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads configuration from a YAML file.

    Args:
        config_path (str | Path | None): Explicit path to a YAML config file.
            If not provided, this function resolves in order:
            1) CONFIG_PATH environment variable
            2) APP_ENV = "prod" -> prod.config.yaml, otherwise dev.config.yaml
            3) Fallback to config.yaml (legacy)

    Returns:
        dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    if config_path is not None:
        path = Path(config_path)
    else:
        config_path_env = os.getenv("CONFIG_PATH")
        if config_path_env:
            path = Path(config_path_env)
        else:
            app_env = os.getenv("APP_ENV", "dev").lower()
            env_config = "prod.config.yaml" if app_env == "prod" else "dev.config.yaml"
            path = Path(env_config)
            if not path.exists():
                path = Path("config.yaml")

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open("r") as f:
        return yaml.safe_load(f)
