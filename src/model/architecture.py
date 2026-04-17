from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertModel

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_BERT_DIR = Path("artifacts/bert")


class BertSentimentClassifier(nn.Module):
    """BERT encoder with a dropout and linear classification head."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        local_model_dir: str | Path | None = None,
    ) -> None:
        """Builds the classifier.

        Args:
            model_name: Hugging Face model id or path (used if no local weights).
            num_labels: Number of output classes.
            dropout: Dropout before the linear layer.
            local_model_dir: If set and contains ``config.json``, loads BERT
                weights from this directory (matches ``artifacts/bert`` layout).
        """
        super().__init__()
        self.num_labels = num_labels
        load_path: str | Path = model_name
        candidates: list[Path] = []
        if local_model_dir is not None:
            candidates.append(Path(local_model_dir))
        candidates.append(DEFAULT_LOCAL_BERT_DIR)
        for cand in candidates:
            if (cand / "config.json").exists():
                load_path = cand
                logger.info("Loading BERT weights from %s", load_path)
                break
        self.bert = BertModel.from_pretrained(str(load_path))
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass.

        Args:
            input_ids: Token ids ``(batch, seq)``.
            attention_mask: Attention mask ``(batch, seq)``.
            labels: Optional gold labels ``(batch,)`` for loss.

        Returns:
            Dict with ``logits`` and optional ``loss`` (cross-entropy).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        logits = self.classifier(x)
        loss: torch.Tensor | None = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
