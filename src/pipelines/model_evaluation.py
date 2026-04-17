from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from src.data.loader import TweetLoader
from src.data.preprocessor import TweetPreprocessor
from src.model.architecture import BertSentimentClassifier
from src.model.evaluate import Evaluator
from src.model.loader import load_checkpoint
from src.utils import (
    get_device,
    load_config,
    resolve_eval_checkpoint_path,
    resolve_split_csv_paths,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Loads a checkpoint and evaluates on the held-out test split."""
    try:
        config = load_config()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    device = get_device()
    print(f"Evaluating model for {config['project']['name']} on {device}...")

    try:
        _train_csv, _val_csv, test_csv = resolve_split_csv_paths(config)
        ckpt_path = resolve_eval_checkpoint_path(config)
    except (KeyError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
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
        test_loader = loader.create_dataloader_from_csv(
            test_csv,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: failed to build test dataloader: {exc}", file=sys.stderr)
        logger.exception("test dataloader failed")
        sys.exit(1)

    model = BertSentimentClassifier(
        model_name=model_cfg["name"],
        num_labels=num_labels,
    )
    try:
        load_checkpoint(model, ckpt_path, device)
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        print(f"Error: failed to load checkpoint {ckpt_path}: {exc}", file=sys.stderr)
        logger.exception("load_checkpoint failed")
        sys.exit(1)

    metrics = Evaluator.evaluate(
        model,
        test_loader,
        device,
        desc="test",
        num_labels=num_labels,
    )
    print(json.dumps(metrics, indent=2))

    out_dir = Path(model_cfg["checkpoint_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
