from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.data.loader import TweetLoader
from src.data.preprocessor import TweetPreprocessor
from src.model.architecture import BertSentimentClassifier
from src.model.train import Trainer
from src.utils import (
    get_device,
    load_config,
    resolve_split_csv_paths,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Orchestrates training: load splits, dataloaders, model, and ``Trainer``."""
    try:
        config = load_config()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    print(f"Starting training for {config['project']['name']} on {device}...")

    try:
        train_csv, val_csv, _test_csv = resolve_split_csv_paths(config)
    except (KeyError, FileNotFoundError) as exc:
        print(f"Error: could not resolve split CSV paths: {exc}", file=sys.stderr)
        sys.exit(1)

    data_cfg = config["data"]
    model_cfg = config["model"]
    batch_size = int(data_cfg["batch_size"])
    max_length = int(data_cfg["max_length"])
    num_labels = int(model_cfg["num_labels"])

    try:
        preprocessor = TweetPreprocessor(
            model_name=model_cfg["name"],
            max_length=max_length,
        )
    except RuntimeError as exc:
        print(f"Error: tokenizer setup failed: {exc}", file=sys.stderr)
        sys.exit(1)

    tokenizer = preprocessor.tokenizer
    loader = TweetLoader(
        text_col="tweet_text",
        target_col="target",
        encoding="utf-8",
    )

    try:
        train_loader = loader.create_dataloader_from_csv(
            train_csv,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=True,
        )
        val_loader = loader.create_dataloader_from_csv(
            val_csv,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: failed to build dataloaders: {exc}", file=sys.stderr)
        logger.exception("create_dataloader_from_csv failed")
        sys.exit(1)

    model = BertSentimentClassifier(
        model_name=model_cfg["name"],
        num_labels=num_labels,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=float(model_cfg["learning_rate"]),
        epochs=int(model_cfg["epochs"]),
        checkpoint_dir=Path(model_cfg["checkpoint_path"]),
        num_labels=num_labels,
    )

    best_path = trainer.fit()
    print(f"Training complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
