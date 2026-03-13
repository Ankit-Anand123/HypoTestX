"""
Data preprocessing / transformation utilities.

All transformations return a NEW list — the original data is never modified.

Provides
--------
standardize(data)             -> z-scores: (x - mean) / std
normalize(data, low, high)    -> scale to [low, high]  (min-max)
winsorize(data, limits)       -> clip outliers at p-th and (1-p)-th percentiles
log_transform(data, base)     -> log(x + shift) transformation
rank_transform(data)          -> integer ranks (1-based, average ties)
center(data)                  -> mean-center: x - mean
robust_scale(data)            -> (x - median) / IQR
apply(data, fn)               -> apply arbitrary fn element-wise
"""

from typing import Callable, List, Optional, Tuple

from ..math.basic import abs_value, ln, sqrt
from ..math.statistics import iqr, mean, median, percentile, std

# ---------------------------------------------------------------------------
# Standardize (z-score)
# ---------------------------------------------------------------------------


def standardize(
    data: List[float],
    ddof: int = 1,
) -> List[float]:
    """
    Standardize data to zero mean and unit variance (z-scores).

    z_i = (x_i - mean) / std

    Parameters
    ----------
    data : numeric list (n >= 2)
    ddof : divisor for standard deviation (default 1 = sample std)

    Returns
    -------
    List[float] of z-scores
    """
    data = [float(x) for x in data]
    n = len(data)
    if n < 2:
        raise ValueError("standardize requires at least 2 data points")
    mu = mean(data)
    s = std(data, ddof=ddof)
    if s == 0:
        raise ValueError("Cannot standardize data with zero variance")
    return [(x - mu) / s for x in data]


# ---------------------------------------------------------------------------
# Normalize (min-max scaling)
# ---------------------------------------------------------------------------


def normalize(
    data: List[float],
    low: float = 0.0,
    high: float = 1.0,
) -> List[float]:
    """
    Scale data to the interval [low, high] using min-max normalization.

    scaled_i = low + (x_i - min) / (max - min) * (high - low)

    Parameters
    ----------
    data : numeric list
    low  : target minimum (default 0)
    high : target maximum (default 1)

    Returns
    -------
    List[float]
    """
    data = [float(x) for x in data]
    if not data:
        raise ValueError("normalize requires non-empty data")
    x_min = min(data)
    x_max = max(data)
    if x_min == x_max:
        # All values equal — return constant 0.5 between low and high
        mid = (low + high) / 2.0
        return [mid] * len(data)
    span = x_max - x_min
    target = high - low
    return [low + (x - x_min) / span * target for x in data]


# ---------------------------------------------------------------------------
# Winsorize
# ---------------------------------------------------------------------------


def winsorize(
    data: List[float],
    limits: Tuple[float, float] = (0.05, 0.05),
) -> List[float]:
    """
    Winsorize data by clipping extreme values at the given percentile limits.

    Parameters
    ----------
    data   : numeric list
    limits : (lower_fraction, upper_fraction) to clip off each tail
             e.g. (0.05, 0.05) clips the bottom and top 5%.

    Returns
    -------
    List[float] with extreme values replaced by the boundary percentiles

    Example
    -------
    >>> winsorize([1, 2, …, 100], limits=(0.10, 0.10))
    # values below p10 become p10; values above p90 become p90
    """
    data = [float(x) for x in data]
    if len(data) < 2:
        raise ValueError("winsorize requires at least 2 data points")
    lo_frac, hi_frac = limits
    if not (0.0 <= lo_frac < 0.5) or not (0.0 <= hi_frac < 0.5):
        raise ValueError("Each limit must be in [0, 0.5)")

    lo_pct = percentile(data, lo_frac * 100.0)
    hi_pct = percentile(data, (1.0 - hi_frac) * 100.0)
    return [max(lo_pct, min(hi_pct, x)) for x in data]


# ---------------------------------------------------------------------------
# Log transform
# ---------------------------------------------------------------------------


def log_transform(
    data: List[float],
    base: str = "natural",
    shift: float = 0.0,
) -> List[float]:
    """
    Apply a logarithmic transformation.

    Parameters
    ----------
    data  : numeric list (all values must be > -shift, i.e. x + shift > 0)
    base  : 'natural' (default, ln), '10' (log base 10), '2' (log base 2)
    shift : constant added before log to handle zero/negative values
            e.g. shift=1 computes log(x + 1)

    Returns
    -------
    List[float]
    """
    data = [float(x) for x in data]
    if any(x + shift <= 0 for x in data):
        raise ValueError(
            f"All (x + shift) values must be > 0 for log transform. "
            f"Consider setting shift > {-min(data):.4f}"
        )

    if base == "natural":
        return [ln(x + shift) for x in data]
    elif base == "10":
        ln10 = ln(10.0)
        return [ln(x + shift) / ln10 for x in data]
    elif base == "2":
        ln2 = ln(2.0)
        return [ln(x + shift) / ln2 for x in data]
    else:
        raise ValueError("base must be 'natural', '10', or '2'")


# ---------------------------------------------------------------------------
# Rank transform
# ---------------------------------------------------------------------------


def rank_transform(
    data: List[float],
    method: str = "average",
) -> List[float]:
    """
    Assign ranks to data values (1-based, handles ties).

    Parameters
    ----------
    data   : numeric list
    method : tie-breaking method
             'average' (default) — tied values receive the average rank
             'min'               — tied values receive the lowest rank
             'max'               — tied values receive the highest rank
             'ordinal'           — values ranked by order of appearance

    Returns
    -------
    List[float] : ranks in the same order as input
    """
    data = [float(x) for x in data]
    n = len(data)
    order = sorted(range(n), key=lambda i: data[i])  # indices sorted by value

    ranks = [0.0] * n

    if method == "ordinal":
        for rank, idx in enumerate(order, start=1):
            ranks[idx] = float(rank)
        return ranks

    i = 0
    while i < n:
        j = i
        while j < n - 1 and data[order[j]] == data[order[j + 1]]:
            j += 1
        # order[i..j] are tied
        if method == "average":
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
        elif method == "min":
            for k in range(i, j + 1):
                ranks[order[k]] = float(i + 1)
        elif method == "max":
            for k in range(i, j + 1):
                ranks[order[k]] = float(j + 1)
        else:
            raise ValueError("method must be 'average', 'min', 'max', or 'ordinal'")
        i = j + 1

    return ranks


# ---------------------------------------------------------------------------
# Center
# ---------------------------------------------------------------------------


def center(data: List[float]) -> List[float]:
    """
    Mean-center data: return x_i - mean(x).
    """
    data = [float(x) for x in data]
    mu = mean(data)
    return [x - mu for x in data]


# ---------------------------------------------------------------------------
# Robust scale
# ---------------------------------------------------------------------------


def robust_scale(data: List[float]) -> List[float]:
    """
    Scale using median and IQR: (x - median) / IQR.
    Robust to outliers.
    """
    data = [float(x) for x in data]
    med = median(data)
    iqa = iqr(data)
    if iqa == 0:
        raise ValueError("IQR is zero; cannot apply robust scaling")
    return [(x - med) / iqa for x in data]


# ---------------------------------------------------------------------------
# Generic apply
# ---------------------------------------------------------------------------


def apply(
    data: List[float],
    fn: Callable[[float], float],
) -> List[float]:
    """
    Apply a function element-wise to a list.

    Parameters
    ----------
    data : numeric list
    fn   : callable(float) -> float

    Returns
    -------
    List[float]

    Example
    -------
    >>> from hypotestx.math.basic import sqrt
    >>> apply([4.0, 9.0, 16.0], sqrt)
    [2.0, 3.0, 4.0]
    """
    return [fn(float(x)) for x in data]


__all__ = [
    "standardize",
    "normalize",
    "winsorize",
    "log_transform",
    "rank_transform",
    "center",
    "robust_scale",
    "apply",
]
