from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.data.loader import TweetLoader
from src.data.preprocessor import TweetPreprocessor
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
    loader = TweetLoader(
        text_col="tweet_text",
        target_col="target"
    )
    try:
        preprocessor = TweetPreprocessor(
            model_name=config["model"]["name"],
            max_length=config["data"]["max_length"]
        )
    except RuntimeError as exc:
        print(f"Error: tokenizer setup failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Also ensure base model is downloaded to artifacts/bert
    try:
        download_and_save_base_model(
            model_name=config["model"]["name"],
            save_path="artifacts/bert"
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

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")

        try:
            # 1. Load data
            df = loader.load_csv(csv_file)

            # 2. Clean data
            df = preprocessor.preprocess_df(df, text_col="tweet_text")

            # 3. Save processed CSV
            output_file = processed_path / f"cleaned_{csv_file.name}"
            df.to_csv(output_file, index=False)
        except (FileNotFoundError, ValueError, KeyError, RuntimeError, OSError) as exc:
            print(f"Skipping {csv_file.name}: {exc}", file=sys.stderr)
            logger.exception("Failed processing %s", csv_file)
            continue

        print(f"Saved cleaned data to {output_file}")

    print("Pre-processing complete.")


if __name__ == "__main__":
    main()
