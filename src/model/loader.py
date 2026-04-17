from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import BertModel

logger = logging.getLogger(__name__)


def download_and_save_base_model(model_name: str, save_path: str | Path) -> None:
    """Downloads the base BERT model and saves it locally.

    Args:
        model_name: The name of the BERT model to download.
        save_path: Directory to save the model.
    """
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    if not (path / "config.json").exists():
        print(f"Downloading base model {model_name} to {path}...")
        model = BertModel.from_pretrained(model_name)
        model.save_pretrained(path)
    else:
        print(f"Base model already exists at {path}.")


def save_checkpoint(
    model: nn.Module,
    save_path: str | Path,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Saves a model checkpoint.

    Args:
        model (BertSentimentClassifier): Initialized model.
        save_path (str | Path): Directory to save the checkpoint.
        epoch (int): Standard epoch number.
        optimizer (torch.optim.Optimizer | None): Optional optimizer state.
    """
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path / f"checkpoint_epoch_{epoch}.pt")


def save_best_model(
    model: nn.Module,
    save_dir: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    val_loss: float | None = None,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Saves the best model weights to ``best_model.pt`` under ``save_dir``.

    Args:
        model: Trained module.
        save_dir: Checkpoint directory.
        optimizer: Optional optimizer state.
        epoch: Epoch index when this snapshot was taken.
        val_loss: Optional validation loss for metadata.
        metrics: Optional scalar metrics to store in the checkpoint.

    Returns:
        Path to the written ``best_model.pt`` file.
    """
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    out = path / "best_model.pt"
    torch.save(payload, out)
    logger.info("Saved best model checkpoint to %s", out)
    return out


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Loads a model weight from a checkpoint.

    Args:
        model (BertSentimentClassifier): Initialized architecture.
        checkpoint_path (str | Path): Path to the .pt file.
        device (torch.device): Device to load the model on.

    Returns:
        BertSentimentClassifier: The model with loaded weights.
    """
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model
