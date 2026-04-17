from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def train_val_test_split_dataframe(
    df: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a DataFrame into train, validation, and test sets.

    Any rows not accounted for by integer indexing of the three fractions
    are included in the test set.

    Args:
        df: Input DataFrame.
        train_fraction: Fraction for training (e.g. 0.7).
        val_fraction: Fraction for validation (e.g. 0.1).
        test_fraction: Fraction for test (e.g. 0.2).
        random_state: Seed for reproducible shuffling.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If fractions do not sum to approximately 1.0 or are invalid.
    """
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-5:
        msg = (
            f"train_fraction + val_fraction + test_fraction must sum to 1.0; "
            f"got {total}"
        )
        raise ValueError(msg)
    if min(train_fraction, val_fraction, test_fraction) < 0:
        raise ValueError("Split fractions must be non-negative")

    if len(df) == 0:
        logger.warning("Empty DataFrame passed to train_val_test_split_dataframe")
        empty = df.iloc[0:0].copy()
        return empty.copy(), empty.copy(), empty.copy()

    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    i_val_end = n_train + n_val
    train_df = shuffled.iloc[:n_train].copy()
    val_df = shuffled.iloc[n_train:i_val_end].copy()
    test_df = shuffled.iloc[i_val_end:].copy()
    return train_df, val_df, test_df


def validate_split_config(data_cfg: dict[str, Any]) -> tuple[float, float, float]:
    """Reads train/val/test fractions from a config ``data`` section.

    Args:
        data_cfg: The ``config['data']`` dictionary.

    Returns:
        Tuple (train_fraction, val_fraction, test_fraction).

    Raises:
        KeyError: If required keys are missing.
        ValueError: If fractions are invalid.
    """
    train_f = float(data_cfg["train_fraction"])
    val_f = float(data_cfg["val_fraction"])
    test_f = float(data_cfg["test_fraction"])
    total = train_f + val_f + test_f
    if abs(total - 1.0) > 1e-5:
        msg = (
            f"train_fraction + val_fraction + test_fraction must sum to 1.0; "
            f"got {total}"
        )
        raise ValueError(msg)
    if min(train_f, val_f, test_f) < 0:
        raise ValueError("Split fractions must be non-negative")
    return train_f, val_f, test_f
