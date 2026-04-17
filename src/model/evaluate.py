from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs model evaluation loops and aggregates classification metrics."""

    @staticmethod
    @torch.inference_mode()
    def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        *,
        desc: str = "eval",
        num_labels: int | None = None,
    ) -> dict[str, float]:
        """Computes average loss and sklearn metrics on a split.

        Args:
            model: Classifier in eval mode is assumed; this method sets ``eval()``.
            dataloader: Batches with ``input_ids``, ``attention_mask``, ``labels``.
            device: Torch device.
            desc: Progress bar description.
            num_labels: Number of classes (for macro metrics); inferred from
                predictions if omitted.

        Returns:
            Dict with ``loss``, ``accuracy``, ``precision_macro``, ``recall_macro``,
            and ``f1_macro``.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_n = 0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out: dict[str, Any] = model(
                input_ids,
                attention_mask,
                labels=None,
            )
            logits = out["logits"]
            loss = criterion(logits, labels)
            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            total_n += batch_n
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

        if total_n == 0:
            logger.warning("Evaluator received an empty dataloader")
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
            }

        avg_loss = total_loss / max(total_n, 1)
        n_classes = num_labels
        if n_classes is None:
            n_classes = (
                max(
                    (max(all_labels) if all_labels else 0),
                    (max(all_preds) if all_preds else 0),
                )
                + 1
            )

        labels_range = list(range(int(n_classes)))
        accuracy = float(accuracy_score(all_labels, all_preds))
        precision, recall, _, _ = precision_recall_fscore_support(
            all_labels,
            all_preds,
            labels=labels_range,
            average="macro",
            zero_division=0,
        )
        macro_f1 = float(
            f1_score(
                all_labels,
                all_preds,
                average="macro",
                zero_division=0,
            )
        )
        metrics = {
            "loss": float(avg_loss),
            "accuracy": accuracy,
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": macro_f1,
        }
        logger.info("Evaluation %s: %s", desc, metrics)
        return metrics
