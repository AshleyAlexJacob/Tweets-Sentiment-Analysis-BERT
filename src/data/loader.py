from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.data.dataset import TweetDataset

logger = logging.getLogger(__name__)

class TweetLoader:
    """Utility class for loading tweet data from CSV files."""

    def __init__(
        self,
        text_col: str = "tweet_text",
        target_col: str = "target",
        encoding: str = "ISO-8859-1",
    ) -> None:
        """Initializes the TweetLoader.

        Args:
            text_col: Name of the column containing tweet text.
            target_col: Name of the column containing sentiment targets.
            encoding: Character encoding of the CSV file.
        """
        self.text_col = text_col
        self.target_col = target_col
        self.encoding = encoding

    def load_csv(self, file_path: str | Path) -> pd.DataFrame:
        """Loads a CSV file and selects the required columns.

        Matches the Sentiment140 format where headers might be missing.
        Columns: target, id, date, flag, user, tweet_text.

        Args:
            file_path: Path to the CSV file.

        Returns:
            A pandas DataFrame containing only the required columns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing after parsing.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        # Try to read with headers first
        try:
            df = pd.read_csv(path, encoding=self.encoding)
            if self.text_col not in df.columns or self.target_col not in df.columns:
                # If headers are missing or columns don't match, reload without headers
                # and assign specific names based on the Sentiment140 schema
                df = pd.read_csv(
                    path,
                    encoding=self.encoding,
                    header=None,
                    names=["target", "id", "date", "flag", "user", "tweet_text"],
                )
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            logger.warning(
                "Primary CSV read failed for %s (%s); trying fallback parse.",
                path,
                exc,
            )
            try:
                df = pd.read_csv(
                    path,
                    encoding=self.encoding,
                    header=None,
                    on_bad_lines="skip",
                )
            except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc2:
                logger.exception("Failed to read CSV: %s", path)
                raise ValueError(f"Could not parse CSV file: {path}") from exc2
            # Assign names if possible
            if len(df.columns) == 6:
                df.columns = ["target", "id", "date", "flag", "user", "tweet_text"]

        try:
            return df[[self.target_col, self.text_col]]
        except KeyError as exc:
            raise ValueError(
                f"CSV must contain columns {self.text_col!r} and {self.target_col!r}; "
                f"got columns: {list(df.columns)}"
            ) from exc

    def create_dataloader(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 32,
        max_length: int = 128,
        shuffle: bool = True,
    ) -> DataLoader:
        """Creates a PyTorch DataLoader from a DataFrame.

        Args:
            df: DataFrame containing the data.
            tokenizer: BERT tokenizer instance.
            batch_size: Number of samples per batch.
            max_length: Maximum sequence length.
            shuffle: Whether to shuffle the data.

        Returns:
            A PyTorch DataLoader.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        try:
            texts = df[self.text_col].tolist()
            targets = df[self.target_col].tolist()
        except KeyError as exc:
            raise ValueError(
                f"DataFrame must contain columns {self.text_col!r} and "
                f"{self.target_col!r}; got: {list(df.columns)}"
            ) from exc

        try:
            dataset = TweetDataset(
                texts=texts,
                targets=targets,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        except Exception as exc:
            logger.exception("Failed to create DataLoader")
            raise RuntimeError("Failed to create DataLoader from DataFrame") from exc
