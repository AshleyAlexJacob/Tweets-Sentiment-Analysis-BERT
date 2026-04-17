from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.data.loader import TweetLoader
from src.data.preprocessor import TweetPreprocessor, remap_sentiment140_targets
from src.data.splits import train_val_test_split_dataframe, validate_split_config
from src.model.loader import download_and_save_base_model
from src.utils import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    """Main pipeline for data preprocessing."""
    try:
        config = load_config()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Preprocessing data for {config['project']['name']}...")

    # Define paths from config
    raw_path = Path(config["data"]["raw_path"])
    processed_path = Path(config["data"]["processed_path"])

    # Create directories if they don't exist
    try:
        raw_path.mkdir(parents=True, exist_ok=True)
        processed_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Error: could not create data directories: {exc}", file=sys.stderr)
        sys.exit(1)

    # Initialize loader and preprocessor
    loader = TweetLoader(text_col="tweet_text", target_col="target")
    try:
        preprocessor = TweetPreprocessor(
            model_name=config["model"]["name"], max_length=config["data"]["max_length"]
        )
    except RuntimeError as exc:
        print(f"Error: tokenizer setup failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Also ensure base model is downloaded to artifacts/bert
    try:
        download_and_save_base_model(
            model_name=config["model"]["name"], save_path="artifacts/bert"
        )
    except Exception:
        print(
            "Error: could not download or save base model. "
            "Check network, Hugging Face access, and HF_TOKEN if rate-limited.",
            file=sys.stderr,
        )
        logger.exception("download_and_save_base_model failed")
        sys.exit(1)

    # Find all CSV files in raw_path
    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_path}. Please place your raw data there.")
        return

    try:
        train_f, val_f, test_f = validate_split_config(config["data"])
    except (KeyError, ValueError, TypeError) as exc:
        print(f"Error: invalid split configuration: {exc}", file=sys.stderr)
        sys.exit(1)
    random_state = int(config["data"].get("random_state", 42))

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")

        try:
            # 1. Load data
            df = loader.load_csv(csv_file)

            # 2. Clean text
            df = preprocessor.preprocess_df(df, text_col="tweet_text")

            # 3. Map Sentiment140 labels to 3 classes; drop others
            df = remap_sentiment140_targets(df, target_col="target")

            # 4. Full cleaned CSV (all mapped rows)
            stem = Path(csv_file.name).stem
            output_file = processed_path / f"cleaned_{csv_file.name}"
            df.to_csv(output_file, index=False)

            # 5. Train / validation / test splits
            train_df, val_df, test_df = train_val_test_split_dataframe(
                df,
                train_f,
                val_f,
                test_f,
                random_state=random_state,
            )
            train_path = processed_path / f"train_{stem}.csv"
            val_path = processed_path / f"val_{stem}.csv"
            test_path = processed_path / f"test_{stem}.csv"
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
        except (FileNotFoundError, ValueError, KeyError, RuntimeError, OSError) as exc:
            print(f"Skipping {csv_file.name}: {exc}", file=sys.stderr)
            logger.exception("Failed processing %s", csv_file)
            continue

        print(f"Saved cleaned data to {output_file}")
        print(
            f"Saved splits: {train_path.name} ({len(train_df)}), "
            f"{val_path.name} ({len(val_df)}), {test_path.name} ({len(test_df)})"
        )

    print("Pre-processing complete.")


if __name__ == "__main__":
    main()
