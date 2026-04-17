from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)


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


def resolve_split_csv_paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Resolves train, validation, and test CSV paths under ``processed_path``.

    If ``data.train_csv``, ``data.val_csv``, and ``data.test_csv`` are set, those
    paths (relative to ``data.processed_path``) are used. Otherwise the first
    ``train_<stem>.csv`` in the processed directory is used, with matching
    ``val_<stem>.csv`` and ``test_<stem>.csv``.

    Args:
        config: Full configuration dict from ``load_config``.

    Returns:
        Tuple of absolute paths ``(train_csv, val_csv, test_csv)``.

    Raises:
        KeyError: If required ``data`` keys are missing.
        FileNotFoundError: If expected CSV files do not exist.
    """
    data = config["data"]
    processed = Path(data["processed_path"])
    train_name = data.get("train_csv")
    val_name = data.get("val_csv")
    test_name = data.get("test_csv")
    if train_name and val_name and test_name:
        train_path = processed / str(train_name)
        val_path = processed / str(val_name)
        test_path = processed / str(test_name)
        missing = [p for p in (train_path, val_path, test_path) if not p.exists()]
        if missing:
            msg = "Configured split CSV not found: " + ", ".join(
                str(p) for p in missing
            )
            raise FileNotFoundError(msg)
        return train_path, val_path, test_path

    candidates = sorted(processed.glob("train_*.csv"))
    if not candidates:
        msg = (
            f"No train_*.csv under {processed}. Run "
            f"`python -m src.pipelines.data_preprocessing` first."
        )
        raise FileNotFoundError(msg)
    train_path = candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Multiple train_*.csv files in %s; using %s",
            processed,
            train_path.name,
        )
    stem = train_path.stem.removeprefix("train_")
    val_path = processed / f"val_{stem}.csv"
    test_path = processed / f"test_{stem}.csv"
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation split not found: {val_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test split not found: {test_path}")
    return train_path, val_path, test_path


def resolve_eval_checkpoint_path(config: dict[str, Any]) -> Path:
    """Resolves the checkpoint file used for evaluation / inference.

    Uses ``model.eval_checkpoint`` when set; otherwise ``model.checkpoint_path``
    / ``best_model.pt``.

    Args:
        config: Full configuration dict.

    Returns:
        Path to the ``.pt`` checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    model_cfg = config["model"]
    explicit = model_cfg.get("eval_checkpoint")
    if explicit:
        path = Path(str(explicit))
        if not path.is_file():
            raise FileNotFoundError(f"eval_checkpoint not found: {path}")
        return path
    ckpt_dir = Path(str(model_cfg["checkpoint_path"]))
    best = ckpt_dir / "best_model.pt"
    if not best.is_file():
        raise FileNotFoundError(
            f"best_model.pt not found under {ckpt_dir}. Train the model first."
        )
    return best
