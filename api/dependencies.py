from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from src.data.preprocessor import TweetPreprocessor
from src.model.architecture import BertSentimentClassifier
from src.model.loader import load_checkpoint
from src.utils import get_device, load_config, resolve_eval_checkpoint_path

logger = logging.getLogger(__name__)


@lru_cache
def get_config() -> dict[str, Any]:
    """Loads and caches the configuration."""
    return load_config()


@lru_cache
def get_preprocessor() -> TweetPreprocessor:
    """Loads and caches the tweet preprocessor."""
    config = get_config()
    return TweetPreprocessor(
        model_name=config["model"]["name"], max_length=config["data"]["max_length"]
    )


@lru_cache
def get_model() -> BertSentimentClassifier:
    """Loads and caches the BERT model."""
    config = get_config()
    device = get_device()

    model = BertSentimentClassifier(
        model_name=config["model"]["name"],
        num_labels=int(config["model"]["num_labels"]),
    )

    try:
        ckpt_path = resolve_eval_checkpoint_path(config)
        load_checkpoint(model, ckpt_path, device)
    except FileNotFoundError:
        logger.warning(
            "No trained checkpoint found (expected best_model.pt under "
            "model.checkpoint_path, or model.eval_checkpoint). Serving with "
            "randomly initialized weights.",
        )

    model.to(device)
    model.eval()
    return model
