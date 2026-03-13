"""
Data utility functions for preparing and inspecting data before testing.

Provides
--------
coerce_numeric(data, name)            -> List[float]   + warnings for bad values
drop_missing(data, *more_cols)        -> cleaned data (removes NaN / None rows)
detect_missing(data)                  -> (n_missing, indices)
group_by(data, labels)                -> dict mapping label -> values
split_groups(*args)                   -> list of numeric groups from mixed input
validate_sample_data(data, min_size)  -> raise or return cleaned list
summary_table(*groups, names)         -> str   one-line stats per group
are_paired(a, b)                      -> bool  (same length, plausibly paired)
"""

from typing import Any, Dict, List, Optional, Tuple

from ..math.statistics import mean, std

# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def coerce_numeric(
    data: List[Any],
    name: str = "data",
    drop_invalid: bool = False,
) -> List[float]:
    """
    Convert a list of mixed-type values to floats.

    Parameters
    ----------
    data         : input list (may contain strings, ints, floats, None, etc.)
    name         : label used in warning messages
    drop_invalid : if True, silently drop non-numeric values;
                   if False (default), raise ValueError on bad values

    Returns
    -------
    List[float]

    Examples
    --------
    >>> coerce_numeric([1, '2.5', None, 3], drop_invalid=True)
    [1.0, 2.5, 3.0]
    """
    result = []
    bad = []
    for i, v in enumerate(data):
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            bad.append(i)
            continue
        try:
            result.append(float(v))
        except (TypeError, ValueError):
            if drop_invalid:
                bad.append(i)
            else:
                raise ValueError(f"{name}: cannot convert value {v!r} at index {i} to float")
    if bad and not drop_invalid:
        pass  # already raised above
    return result


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------


def detect_missing(data: List[Any]) -> Tuple[int, List[int]]:
    """
    Detect None and float NaN entries.

    Returns
    -------
    (n_missing, missing_indices)
    """
    indices = []
    for i, v in enumerate(data):
        if v is None:
            indices.append(i)
        elif isinstance(v, float) and v != v:  # NaN check
            indices.append(i)
    return len(indices), indices


def drop_missing(
    *columns: List[Any],
) -> Tuple[List[List[float]], int]:
    """
    Remove rows where ANY column has a missing value (None or NaN).

    Parameters
    ----------
    *columns : one or more lists of equal length

    Returns
    -------
    (cleaned_columns, n_dropped)
        cleaned_columns : list of cleaned lists (same number as input)
        n_dropped       : number of rows removed

    Example
    -------
    >>> a, b, n = drop_missing([1, None, 3], [4, 5, 6])
    >>> # a=[1.0, 3.0], b=[4.0, 6.0], n=1
    """
    if not columns:
        return [], 0

    # Validate all same length
    lengths = [len(c) for c in columns]
    if len(set(lengths)) != 1:
        raise ValueError("All columns must have the same length")

    n_rows = lengths[0]
    keep = []
    n_drop = 0
    for i in range(n_rows):
        row_missing = False
        for col in columns:
            v = col[i]
            if v is None or (isinstance(v, float) and v != v):
                row_missing = True
                break
        if row_missing:
            n_drop += 1
        else:
            keep.append(i)

    cleaned = [[float(col[i]) for i in keep] for col in columns]
    return cleaned, n_drop


# ---------------------------------------------------------------------------
# Group operations
# ---------------------------------------------------------------------------


def group_by(
    data: List[float],
    labels: List[Any],
) -> Dict[Any, List[float]]:
    """
    Split a flat data list into groups based on a parallel labels list.

    Parameters
    ----------
    data   : numeric values
    labels : group labels (same length as data)

    Returns
    -------
    dict mapping each unique label to its observations

    Example
    -------
    >>> group_by([1,2,3,4], ['a','b','a','b'])
    {'a': [1.0, 3.0], 'b': [2.0, 4.0]}
    """
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")

    groups: Dict[Any, List[float]] = {}
    for v, lbl in zip(data, labels):
        groups.setdefault(lbl, []).append(float(v))
    return groups


def split_groups(*args) -> List[List[float]]:
    """
    Accept either:
      - multiple list arguments  -> return them as-is (after coercion)
      - a single list-of-lists   -> unpack and return

    Useful for normalising input in tests that accept *groups.
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        inner = args[0]
        if inner and isinstance(inner[0], (list, tuple)):
            return [list(map(float, g)) for g in inner]
    return [list(map(float, g)) for g in args]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_sample_data(
    data: List[Any],
    min_size: int = 2,
    name: str = "data",
    allow_missing: bool = False,
) -> List[float]:
    """
    Coerce, optionally clean, and size-check a data list.

    Parameters
    ----------
    data          : input list
    min_size      : minimum acceptable length after cleaning
    name          : label for error messages
    allow_missing : if True, drop missing values silently

    Returns
    -------
    List[float]
    """
    if allow_missing:
        cleaned = coerce_numeric(data, name=name, drop_invalid=True)
    else:
        cleaned = coerce_numeric(data, name=name, drop_invalid=False)

    if len(cleaned) < min_size:
        raise ValueError(f"{name}: need at least {min_size} valid observations, got {len(cleaned)}")
    return cleaned


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def summary_table(
    *groups: List[float],
    names: Optional[List[str]] = None,
) -> str:
    """
    Quick one-line per group stats table: n, mean, std, min, max.

    Parameters
    ----------
    *groups : numeric lists
    names   : optional group labels

    Returns
    -------
    str
    """
    if names is None:
        names = [f"Group {i + 1}" for i in range(len(groups))]

    col_w = max(max(len(n) for n in names), 8) + 2
    header = f"{'Group':<{col_w}}" f"{'n':>6}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}"
    sep = "-" * len(header)
    rows = [header, sep]

    for name, g in zip(names, groups):
        g = [float(x) for x in g]
        n = len(g)
        rows.append(
            f"{name:<{col_w}}"
            f"{n:>6}"
            f"{mean(g):>10.4f}"
            f"{(std(g) if n >= 2 else 0.0):>10.4f}"
            f"{min(g):>10.4f}"
            f"{max(g):>10.4f}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Paired-data check
# ---------------------------------------------------------------------------


def are_paired(a: List[Any], b: List[Any]) -> bool:
    """
    Return True if a and b have the same length (necessary condition for pairing).
    """
    return len(a) == len(b) and len(a) >= 2


__all__ = [
    "coerce_numeric",
    "detect_missing",
    "drop_missing",
    "group_by",
    "split_groups",
    "validate_sample_data",
    "summary_table",
    "are_paired",
]
