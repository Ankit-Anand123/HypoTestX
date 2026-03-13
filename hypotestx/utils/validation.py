"""
Input validation utilities for HypoTestX.

Provides functions for validating dataframes (dict/pandas/polars),
column existence, and numeric/categorical data checks.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_pandas_df(df: Any) -> bool:
    try:
        import pandas as pd

        return isinstance(df, pd.DataFrame)
    except ImportError:
        return False


def _is_polars_df(df: Any) -> bool:
    try:
        import polars as pl

        return isinstance(df, pl.DataFrame)
    except ImportError:
        return False


def _get_columns(df: Any) -> List[str]:
    if isinstance(df, dict):
        return list(df.keys())
    if _is_pandas_df(df) or _is_polars_df(df):
        return list(df.columns)
    raise TypeError(f"Unsupported dataframe type: {type(df)}")


def _get_column_data(df: Any, col: str) -> List[Any]:
    if isinstance(df, dict):
        return list(df[col])
    if _is_pandas_df(df):
        return df[col].tolist()
    if _is_polars_df(df):
        return df[col].to_list()
    raise TypeError(f"Unsupported dataframe type: {type(df)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_dataframe(df: Any) -> None:
    """
    Assert that *df* is a supported dataframe-like object (dict, pandas,
    polars).  Raises :exc:`TypeError` if not.  Raises :exc:`ValueError`
    if the dataframe is empty.
    """
    if df is None:
        raise TypeError("Data must not be None")
    if isinstance(df, dict):
        if not df:
            raise ValueError("Data dict is empty (no columns)")
        lengths = [len(v) for v in df.values()]
        if not lengths or lengths[0] == 0:
            raise ValueError("Data dict has zero rows")
        if len(set(lengths)) > 1:
            raise ValueError(f"Data dict columns have inconsistent lengths: {lengths}")
        return
    if _is_pandas_df(df) or _is_polars_df(df):
        if len(df) == 0:
            raise ValueError("DataFrame has zero rows")
        if len(_get_columns(df)) == 0:
            raise ValueError("DataFrame has no columns")
        return
    raise TypeError(
        f"Unsupported data type: {type(df).__name__}. "
        "Expected dict, pandas.DataFrame, or polars.DataFrame."
    )


def validate_columns(df: Any, *columns: str) -> None:
    """
    Assert that every column in *columns* exists in *df*.

    Raises :exc:`KeyError` listing all missing columns.
    """
    available = set(_get_columns(df))
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(
            f"Column(s) not found in data: {missing}. "
            f"Available columns: {sorted(available)}"
        )


def validate_numeric_column(df: Any, col: str) -> None:
    """
    Assert that *col* contains numeric (int/float) values.

    Raises :exc:`ValueError` if the column cannot be treated as numeric.
    """
    validate_columns(df, col)
    data = _get_column_data(df, col)
    if not data:
        raise ValueError(f"Column '{col}' is empty")
    non_numeric = [v for v in data if not isinstance(v, (int, float)) or v != v]
    # filter out NaN properly
    non_numeric = [v for v in data if not isinstance(v, (int, float))]
    if non_numeric:
        example = non_numeric[:3]
        raise ValueError(f"Column '{col}' contains non-numeric values: {example}")


def validate_categorical_column(df: Any, col: str) -> None:
    """
    Assert that *col* exists.  This is a permissive check that accepts any
    type; downstream tests will handle the actual dtype requirements.
    """
    validate_columns(df, col)


def validate_sample_size(
    data: Sequence[Any], min_size: int = 2, label: str = "data"
) -> None:
    """
    Assert that *data* has at least *min_size* observations.

    Raises :exc:`ValueError` otherwise.
    """
    n = len(data)
    if n < min_size:
        raise ValueError(f"{label} must have at least {min_size} observations, got {n}")


def validate_alpha(alpha: float) -> None:
    """Assert that *alpha* is in the open interval (0, 1)."""
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")


def validate_probability(p: float, name: str = "probability") -> None:
    """Assert that *p* is in [0, 1]."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {p}")


def validate_alternative(alternative: str) -> None:
    """Assert that *alternative* is one of the accepted strings."""
    valid = {"two-sided", "greater", "less"}
    if alternative not in valid:
        raise ValueError(f"alternative must be one of {valid}, got '{alternative}'")


__all__ = [
    "validate_dataframe",
    "validate_columns",
    "validate_numeric_column",
    "validate_categorical_column",
    "validate_sample_size",
    "validate_alpha",
    "validate_probability",
    "validate_alternative",
]
