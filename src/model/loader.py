from __future__ import annotations

import torch
from pathlib import Path

def save_checkpoint(
    model: BertSentimentClassifier,
    save_path: str | Path,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None
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

def load_checkpoint(
    model: BertSentimentClassifier,
    checkpoint_path: str | Path,
    device: torch.device
) -> BertSentimentClassifier:
    """Loads a model weight from a checkpoint.

    Args:
        model (BertSentimentClassifier): Initialized architecture.
        checkpoint_path (str | Path): Path to the .pt file.
        device (torch.device): Device to load the model on.

    Returns:
        BertSentimentClassifier: The model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model
