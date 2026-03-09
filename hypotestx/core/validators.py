"""
Input validation utilities for HypoTestX
"""
from typing import List, Optional, Any

from .exceptions import (
    InsufficientDataError,
    InvalidAlternativeError,
    DataFormatError,
)


def validate_data(data: Any, min_size: int = 2, name: str = "data") -> List[float]:
    """
    Validate that data is a non-empty numeric iterable.

    Args:
        data: Input data (list, tuple, or anything iterable with numeric values)
        min_size: Minimum required number of data points
        name: Name of the parameter (used in error messages)

    Returns:
        List[float] copy of the validated data

    Raises:
        InsufficientDataError: If data is None or has fewer than min_size elements
        DataFormatError: If data cannot be converted to a numeric list
    """
    if data is None:
        raise InsufficientDataError(f"'{name}' cannot be None")

    # Attempt conversion to list
    if isinstance(data, (list, tuple)):
        data_list = list(data)
    else:
        try:
            data_list = list(data)
        except TypeError:
            raise DataFormatError(
                f"'{name}' must be a list or iterable, got {type(data).__name__}"
            )

    if len(data_list) < min_size:
        raise InsufficientDataError(
            f"'{name}' must have at least {min_size} data point(s), got {len(data_list)}"
        )

    validate_numeric(data_list, name)
    return data_list


def validate_numeric(data: List[Any], name: str = "data") -> List[float]:
    """
    Check that every element in data is a real number.

    Args:
        data: Sequence of values to check
        name: Parameter name for error messages

    Returns:
        The same list (converted to float)

    Raises:
        DataFormatError: If any element is not numeric or is NaN/Inf
    """
    result = []
    for i, val in enumerate(data):
        if not isinstance(val, (int, float)):
            raise DataFormatError(
                f"'{name}[{i}]' must be numeric (int or float), got {type(val).__name__}"
            )
        float_val = float(val)
        if float_val != float_val:  # NaN check (NaN != NaN)
            raise DataFormatError(f"'{name}[{i}]' contains NaN")
        if float_val in (float("inf"), float("-inf")):
            raise DataFormatError(f"'{name}[{i}]' contains Infinity")
        result.append(float_val)
    return result


def validate_alpha(alpha: float) -> float:
    """
    Validate significance level.

    Args:
        alpha: Significance level to validate

    Returns:
        Validated alpha value

    Raises:
        TypeError: If alpha is not numeric
        ValueError: If alpha is not in (0, 1)
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be numeric, got {type(alpha).__name__}")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {alpha}")
    return alpha


def validate_alternative(
    alternative: str,
    valid: Optional[List[str]] = None,
) -> str:
    """
    Validate alternative hypothesis string.

    Args:
        alternative: The alternative hypothesis specifier
        valid: Allowed values (defaults to ['two-sided', 'greater', 'less'])

    Returns:
        The validated alternative string

    Raises:
        InvalidAlternativeError: If alternative is not in valid
    """
    if valid is None:
        valid = ["two-sided", "greater", "less"]
    if alternative not in valid:
        raise InvalidAlternativeError(alternative, valid)
    return alternative


def validate_two_groups(
    group1: Any,
    group2: Any,
    min_size: int = 2,
) -> tuple:
    """
    Validate two independent group samples.

    Returns:
        Tuple (list[float], list[float]) of validated groups

    Raises:
        InsufficientDataError or DataFormatError on invalid input
    """
    group1 = validate_data(group1, min_size, "group1")
    group2 = validate_data(group2, min_size, "group2")
    return group1, group2


def validate_paired_data(x: Any, y: Any) -> tuple:
    """
    Validate two paired equal-length samples.

    Returns:
        Tuple (list[float], list[float])

    Raises:
        DataFormatError: If lengths differ
        InsufficientDataError or DataFormatError on invalid input
    """
    x = validate_data(x, 2, "x")
    y = validate_data(y, 2, "y")
    if len(x) != len(y):
        raise DataFormatError(
            f"Paired data must have equal length, got len(x)={len(x)} and len(y)={len(y)}"
        )
    return x, y


def validate_contingency_table(table: Any) -> List[List[float]]:
    """
    Validate a 2-D contingency table.

    Args:
        table: 2-D list (or list of lists) of non-negative counts

    Returns:
        The validated table as List[List[float]]

    Raises:
        DataFormatError: On structural or value errors
    """
    if not table or not isinstance(table, (list, tuple)):
        raise DataFormatError("Contingency table must be a non-empty list of rows")

    nrows = len(table)
    if nrows < 2:
        raise DataFormatError(
            f"Contingency table must have at least 2 rows, got {nrows}"
        )

    ncols = len(table[0]) if table else 0
    if ncols < 2:
        raise DataFormatError(
            f"Contingency table must have at least 2 columns, got {ncols}"
        )

    validated = []
    for r, row in enumerate(table):
        if not isinstance(row, (list, tuple)) or len(row) != ncols:
            raise DataFormatError(
                f"Every row in the contingency table must have {ncols} columns; "
                f"row {r} has {len(row)}"
            )
        validated_row = []
        for c, val in enumerate(row):
            if not isinstance(val, (int, float)) or float(val) < 0:
                raise DataFormatError(
                    f"Contingency table cell [{r}][{c}] must be a non-negative number, got {val}"
                )
            validated_row.append(float(val))
        validated.append(validated_row)

    return validated


def validate_groups(*groups: Any, min_size: int = 2, min_groups: int = 2) -> List[List[float]]:
    """
    Validate three or more independent group samples (e.g., for ANOVA, Kruskal-Wallis).

    Args:
        *groups: Variable number of group data arrays
        min_size: Minimum required size per group
        min_groups: Minimum required number of groups

    Returns:
        List of validated group lists

    Raises:
        InsufficientDataError: If fewer than min_groups groups are provided
    """
    if len(groups) < min_groups:
        raise InsufficientDataError(
            f"At least {min_groups} groups are required, got {len(groups)}"
        )
    return [validate_data(g, min_size, f"group{i + 1}") for i, g in enumerate(groups)]
