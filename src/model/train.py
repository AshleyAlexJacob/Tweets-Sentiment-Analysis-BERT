from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.evaluate import Evaluator
from src.model.loader import save_best_model, save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """Fine-tunes ``BertSentimentClassifier`` with AdamW and validation tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float,
        epochs: int,
        checkpoint_dir: str | Path,
        num_labels: int,
    ) -> None:
        """Initializes the trainer.

        Args:
            model: Sentiment classifier.
            train_loader: Training ``DataLoader``.
            val_loader: Validation ``DataLoader``.
            device: Torch device.
            learning_rate: AdamW learning rate.
            epochs: Number of training epochs.
            checkpoint_dir: Directory for ``checkpoint_epoch_*.pt`` and
                ``best_model.pt``.
            num_labels: Number of classes (passed to evaluator).
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = int(epochs)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.num_labels = num_labels
        self._best_val_loss: float | None = None

    def train_epoch(self) -> float:
        """Runs one training epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_n = 0
        for batch in tqdm(self.train_loader, desc="train", leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            out: dict[str, Any] = self.model(
                input_ids,
                attention_mask,
                labels=labels,
            )
            loss = out["loss"]
            if loss is None:
                raise RuntimeError("Model returned no loss during training")
            loss.backward()
            self.optimizer.step()
            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            total_n += batch_n
        return total_loss / max(total_n, 1)

    def validate(self) -> dict[str, float]:
        """Runs validation and returns metrics."""
        return Evaluator.evaluate(
            self.model,
            self.val_loader,
            self.device,
            desc="val",
            num_labels=self.num_labels,
        )

    def fit(self) -> Path:
        """Trains for ``self.epochs`` and saves per-epoch and best checkpoints.

        Returns:
            Path to ``best_model.pt``.
        """
        best_path: Path | None = None
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            val_loss = val_metrics["loss"]
            logger.info(
                "Epoch %s/%s train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                val_metrics.get("accuracy", 0.0),
            )
            save_checkpoint(
                self.model,
                self.checkpoint_dir,
                epoch,
                self.optimizer,
            )
            if self._best_val_loss is None or val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                best_path = save_best_model(
                    self.model,
                    self.checkpoint_dir,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    metrics=val_metrics,
                )
        if best_path is None:
            raise RuntimeError("Training produced no best checkpoint path")
        return best_path
