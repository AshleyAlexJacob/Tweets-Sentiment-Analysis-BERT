from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from src.data.cleaner import TweetCleaner

logger = logging.getLogger(__name__)


class TweetPreprocessor:
    """Preprocess tweets: cleaning and BERT tokenization."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
    ) -> None:
        """Initializes the TweetPreprocessor.

        Args:
            model_name: The name of the BERT model to use for tokenization.
            max_length: The maximum sequence length for tokenization.

        Raises:
            RuntimeError: If tokenizer cannot be loaded or saved.
        """
        # Ensure artifacts directory exists
        self.save_path = Path("artifacts/bert")
        try:
            self.save_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception("Could not create tokenizer directory %s", self.save_path)
            raise RuntimeError(
                f"Could not create tokenizer directory: {self.save_path}"
            ) from exc

        # Load from local if exists, otherwise download and save
        try:
            if (self.save_path / "vocab.txt").exists():
                self.tokenizer = BertTokenizer.from_pretrained(str(self.save_path))
            else:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.tokenizer.save_pretrained(self.save_path)
        except OSError as exc:
            logger.exception("Tokenizer filesystem error for model %s", model_name)
            raise RuntimeError(
                f"Tokenizer load/save failed (check disk space and paths): {model_name}"
            ) from exc
        except Exception as exc:
            logger.exception("Tokenizer init failed for model %s", model_name)
            raise RuntimeError(
                f"Could not initialize tokenizer for {model_name!r}. "
                "If downloading from Hugging Face, check network and HF_TOKEN."
            ) from exc

        self.max_length = max_length
        self.cleaner = TweetCleaner()

    def clean_text(self, text: str) -> str:
        """Cleans a single tweet text using TweetCleaner.

        Args:
            text: The original tweet text.

        Returns:
            The cleaned tweet text.
        """
        return self.cleaner.clean_tweet(text)

    def preprocess_df(
        self,
        df: pd.DataFrame,
        text_col: str = "tweet_text",
    ) -> pd.DataFrame:
        """Cleans the text column in a DataFrame in place.

        Args:
            df: The DataFrame to process.
            text_col: The name of the column containing tweet text.

        Returns:
            The DataFrame with updated text column.

        Raises:
            KeyError: If ``text_col`` is not in the DataFrame.
            RuntimeError: If cleaning fails unexpectedly.
        """
        try:
            _ = df[text_col]
        except KeyError as exc:
            raise KeyError(
                f"DataFrame has no column {text_col!r}; columns: {list(df.columns)}"
            ) from exc

        try:
            df[text_col] = df[text_col].apply(self.clean_text)
        except TypeError:
            raise
        except Exception as exc:
            logger.exception("Failed while cleaning column %r", text_col)
            raise RuntimeError(f"Failed to preprocess column {text_col!r}") from exc

        return df

    def tokenize_text(self, text: str) -> BatchEncoding:
        """Tokenizes a single text for BERT.

        Args:
            text: The text to tokenize.

        Returns:
            Tokenization results (input_ids, attention_mask, etc.).

        Raises:
            RuntimeError: If tokenization fails.
        """
        try:
            return self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
        except Exception as exc:
            logger.exception("tokenize_text failed (preview: %r)", text[:80])
            raise RuntimeError("Tokenization failed for input text") from exc
