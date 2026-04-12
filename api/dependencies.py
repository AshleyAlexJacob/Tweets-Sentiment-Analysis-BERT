from __future__ import annotations

from functools import lru_cache
from typing import Any
from src.utils import load_config, get_device
from src.model.architecture import BertSentimentClassifier
from src.model.loader import load_checkpoint
from src.data.preprocessor import TweetPreprocessor

@lru_cache()
def get_config() -> dict[str, Any]:
    """Loads and caches the configuration."""
    return load_config()

@lru_cache()
def get_preprocessor() -> TweetPreprocessor:
    """Loads and caches the tweet preprocessor."""
    config = get_config()
    return TweetPreprocessor(
        model_name=config["model"]["name"],
        max_length=config["data"]["max_length"]
    )

@lru_cache()
def get_model() -> BertSentimentClassifier:
    """Loads and caches the BERT model."""
    config = get_config()
    device = get_device()
    
    model = BertSentimentClassifier(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"]
    )
    
    checkpoint_path = config["model"].get("checkpoint_path")
    if checkpoint_path:
        # Note: In a real scenario, we'd check if the file exist
        # model = load_checkpoint(model, checkpoint_path, device)
        pass
        
    model.to(device)
    model.eval()
    return model
