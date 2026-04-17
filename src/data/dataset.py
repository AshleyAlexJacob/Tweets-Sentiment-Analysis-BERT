from __future__ import annotations

import logging

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TweetDataset(Dataset):
    """PyTorch Dataset for tweets and their sentiment targets."""

    def __init__(
        self,
        texts: list[str],
        targets: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ) -> None:
        """Initializes the TweetDataset.

        Args:
            texts: List of tweet texts.
            targets: List of sentiment targets.
            tokenizer: BERT tokenizer instance.
            max_length: Maximum sequence length for tokenization.
        """
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A dictionary containing input_ids, attention_mask, and labels.

        Raises:
            RuntimeError: If tokenization fails.
        """
        text = str(self.texts[idx])
        raw_target = self.targets[idx]
        try:
            target = int(raw_target)
        except (TypeError, ValueError) as exc:
            msg = f"Invalid label at index {idx}: {raw_target!r}"
            raise ValueError(msg) from exc

        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
        except Exception as exc:
            logger.exception(
                "Tokenization failed at index %s (text preview: %r)",
                idx,
                text[:80],
            )
            msg = f"Tokenization failed for sample at index {idx}"
            raise RuntimeError(msg) from exc

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(target, dtype=torch.long),
        }
