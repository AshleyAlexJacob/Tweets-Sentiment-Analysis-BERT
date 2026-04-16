from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class TweetCleaner:
    """A utility class for cleaning tweet text data."""

    def __init__(self) -> None:
        """Initialize the TweetCleaner with compiled regular expressions."""
        # http(s) URLs, including broken schemes like "https: //" (spaces after ':')
        self._url_pattern = re.compile(r"https?\s*:\s*/\s*/\S*", re.IGNORECASE)

        # Parenthetical GMT offsets, e.g. "(GMT +1)", "(GMT+1)", "(GMT -5:30)"
        self._gmt_pattern = re.compile(
            r"\(\s*GMT\s*[\+\-]?\s*\d+(?:\s*:\s*\d+)?\s*\)",
            re.IGNORECASE,
        )

        # Match usernames starting with @
        self._user_pattern = re.compile(r"@\w+")

        # Match variable sequences of '0's
        self._zeros_pattern = re.compile(r"0+")

        # Match emojis and other non-ASCII characters (often unwanted)
        self._emoji_pattern = re.compile(r"[^\x00-\x7F]+")

        # Match multiple spaces to squash them into a single space
        self._spaces_pattern = re.compile(r"\s+")

    def clean_tweet(self, text: str) -> str:
        """Cleans a single tweet text by applying multiple rules.

        Rules applied:
        - Removes HTTP(S) URLs, including malformed ``https: //`` spacing.
        - Removes parenthetical GMT offsets (e.g. ``(GMT +1)``).
        - Masks usernames (e.g., @user to [USER]).
        - Removes variable sequences of '0's.
        - Removes emojis and other non-ASCII unwanted characters.
        - Removes multiple spaces in the middle of the text.
        - Strips leading and trailing whitespace.

        Args:
            text: The original tweet text to clean.

        Returns:
            The cleaned tweet text.

        Raises:
            TypeError: If the input text is not a string.
            ValueError: If cleaning fails unexpectedly.
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected a string, but got {type(text)}")

        try:
            # 1. Remove URLs (handles "https://", "http://", and "https: //" etc.)
            cleaned = self._url_pattern.sub("", text)

            # 2. Remove GMT timezone parentheticals
            cleaned = self._gmt_pattern.sub("", cleaned)

            # 3. Mask usernames starting with @
            cleaned = self._user_pattern.sub("[USER]", cleaned)

            # 4. Remove emojis and non-ASCII characters
            cleaned = self._emoji_pattern.sub("", cleaned)

            # 5. Remove sequences of "0"s
            cleaned = self._zeros_pattern.sub("", cleaned)

            # 6. Squash multiple spaces into a single space
            cleaned = self._spaces_pattern.sub(" ", cleaned)

            # 7. Strip leading and trailing whitespace
            return cleaned.strip()
        except Exception as exc:
            logger.exception("Tweet cleaning failed for input preview: %r", text[:80])
            raise ValueError("Failed to clean tweet text") from exc
