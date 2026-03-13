"""
Comprehensive descriptive statistics.

Provides a single describe() function that returns a DescriptiveStats object,
plus individual convenience functions.  All computation uses the custom
math layer (no numpy/scipy).

Main entry points
-----------------
describe(data, name)           -> DescriptiveStats
five_number_summary(data)      -> dict
detect_outliers(data, method)  -> (indices, values, dict)
frequency_table(data)          -> list of (value, count, pct) tuples
"""

from typing import Any, Dict, List, Optional, Tuple

from ..math.basic import abs_value, ln, sqrt
from ..math.statistics import (
    iqr,
    kurtosis,
    mad,
    mean,
    median,
    mode,
    percentile,
    quartiles,
    range_stat,
    skewness,
    std,
    trimmed_mean,
    variance,
)

# ---------------------------------------------------------------------------
# DescriptiveStats container
# ---------------------------------------------------------------------------


class DescriptiveStats:
    """
    Container for full descriptive statistics of a numeric sample.

    Attributes
    ----------
    n, mean, median, mode, std, variance, min, max, range,
    q1, q2, q3, iqr, mad, skewness, kurtosis,
    sem, cv, trimmed_mean (10%)
    """

    def __init__(self, data: List[float], name: str = "data"):
        self.name = name
        data = sorted(float(x) for x in data)
        n = len(data)
        if n == 0:
            raise ValueError("Cannot compute descriptive statistics on empty data")

        self.n = n
        self.data = data

        self.mean = mean(data)
        self.median = median(data)

        _mode = mode(data)
        self.mode = _mode[0] if len(_mode) == 1 else _mode  # scalar if unimodal

        self.std = std(data) if n >= 2 else 0.0
        self.variance = variance(data) if n >= 2 else 0.0
        self.sem = self.std / sqrt(n) if n >= 2 else 0.0

        self.min = data[0]
        self.max = data[-1]
        self.range = range_stat(data)

        self.q1, self.q2, self.q3 = quartiles(data)
        self.iqr = iqr(data)
        self.mad = mad(data)

        self.skewness = skewness(data) if n >= 3 else 0.0
        self.kurtosis = kurtosis(data) if n >= 4 else 0.0  # excess kurtosis

        self.cv = (self.std / self.mean * 100.0) if self.mean != 0 else float("nan")
        self.trimmed_mean = trimmed_mean(data, 0.10) if n >= 4 else self.mean

        # Percentiles
        self.p05 = percentile(data, 5)
        self.p95 = percentile(data, 95)

    # ------------------------------------------------------------------
    def summary(self, verbose: bool = False) -> str:
        """Return a formatted display string."""
        lines = [
            f"Descriptive Statistics: {self.name}",
            "-" * 38,
            f"  n                 : {self.n}",
            f"  mean              : {self.mean:.4f}",
            f"  std               : {self.std:.4f}",
            f"  SEM               : {self.sem:.4f}",
            f"  variance          : {self.variance:.4f}",
            (
                f"  CV (%)            : {self.cv:.2f}"
                if self.cv == self.cv
                else "  CV (%)            : N/A"
            ),
            f"  min               : {self.min:.4f}",
            f"  Q1 (25%)          : {self.q1:.4f}",
            f"  median            : {self.median:.4f}",
            f"  Q3 (75%)          : {self.q3:.4f}",
            f"  max               : {self.max:.4f}",
            f"  IQR               : {self.iqr:.4f}",
            f"  range             : {self.range:.4f}",
        ]
        if verbose:
            lines += [
                f"  MAD               : {self.mad:.4f}",
                f"  skewness          : {self.skewness:.4f}",
                f"  excess kurtosis   : {self.kurtosis:.4f}",
                f"  trimmed mean(10%) : {self.trimmed_mean:.4f}",
                f"  5th percentile    : {self.p05:.4f}",
                f"  95th percentile   : {self.p95:.4f}",
            ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export all statistics as a dictionary."""
        return {
            "name": self.name,
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "std": self.std,
            "variance": self.variance,
            "sem": self.sem,
            "cv": self.cv,
            "min": self.min,
            "q1": self.q1,
            "q2": self.q2,
            "q3": self.q3,
            "max": self.max,
            "iqr": self.iqr,
            "range": self.range,
            "mad": self.mad,
            "skewness": self.skewness,
            "excess_kurtosis": self.kurtosis,
            "trimmed_mean_10pct": self.trimmed_mean,
            "p05": self.p05,
            "p95": self.p95,
        }

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"DescriptiveStats(name='{self.name}', n={self.n}, "
            f"mean={self.mean:.4f}, std={self.std:.4f})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def describe(
    data: List[float],
    name: str = "data",
    verbose: bool = False,
) -> DescriptiveStats:
    """
    Compute full descriptive statistics for numeric data.

    Parameters
    ----------
    data    : numeric list
    name    : label used in the summary string
    verbose : if True, also prints the summary immediately

    Returns
    -------
    DescriptiveStats object
    """
    ds = DescriptiveStats(data, name=name)
    if verbose:
        print(ds.summary(verbose=True))
    return ds


# ---------------------------------------------------------------------------
# Five-number summary
# ---------------------------------------------------------------------------


def five_number_summary(data: List[float]) -> Dict[str, float]:
    """
    Return {min, Q1, median, Q3, max}.

    Parameters
    ----------
    data : numeric list (at least 2 values)

    Returns
    -------
    dict with keys: 'min', 'q1', 'median', 'q3', 'max'
    """
    data = [float(x) for x in data]
    if len(data) < 2:
        raise ValueError("five_number_summary requires at least 2 data points")
    q1, q2, q3 = quartiles(data)
    return {
        "min": min(data),
        "q1": q1,
        "median": q2,
        "q3": q3,
        "max": max(data),
    }


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


def detect_outliers(
    data: List[float],
    method: str = "iqr",
    threshold: float = 1.5,
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """
    Identify outliers using IQR or Z-score method.

    Parameters
    ----------
    data      : numeric list
    method    : 'iqr' (default) or 'zscore'
    threshold : IQR multiplier (default 1.5, use 3.0 for extreme outliers)
                or Z-score threshold (default interpreted as 1.5*IQR or 3 sigma)

    Returns
    -------
    (indices, values, meta)
        indices : list of 0-based positions of outlier observations
        values  : list of outlier values
        meta    : dict with bounds or thresholds used
    """
    data = [float(x) for x in data]
    n = len(data)
    method = method.lower()

    if method == "iqr":
        q1, _, q3 = quartiles(data)
        _iqr = q3 - q1
        lower = q1 - threshold * _iqr
        upper = q3 + threshold * _iqr
        indices = [i for i, x in enumerate(data) if x < lower or x > upper]
        meta = {
            "lower_bound": lower,
            "upper_bound": upper,
            "iqr": _iqr,
            "threshold": threshold,
        }

    elif method == "zscore":
        if threshold == 1.5:
            threshold = 3.0  # sensible default for z-score
        mu = mean(data)
        s = std(data)
        if s == 0:
            return [], [], {"mean": mu, "std": 0, "threshold": threshold}
        z = [(x - mu) / s for x in data]
        indices = [i for i, zi in enumerate(z) if abs_value(zi) > threshold]
        meta = {"mean": mu, "std": s, "threshold": threshold}

    else:
        raise ValueError("method must be 'iqr' or 'zscore'")

    values = [data[i] for i in indices]
    return indices, values, meta


# ---------------------------------------------------------------------------
# Frequency table
# ---------------------------------------------------------------------------


def frequency_table(
    data: List[float],
) -> List[Tuple[float, int, float]]:
    """
    Build a frequency table for discrete/categorical numeric data.

    Returns
    -------
    List of (value, count, percentage) tuples sorted by value.
    """
    from collections import Counter

    n = len(data)
    if n == 0:
        return []
    counts = Counter(data)
    table = [(v, c, 100.0 * c / n) for v, c in sorted(counts.items())]
    return table


# ---------------------------------------------------------------------------
# Comparison helper (compare multiple groups side-by-side)
# ---------------------------------------------------------------------------


def compare_groups(
    *groups: List[float],
    names: Optional[List[str]] = None,
) -> str:
    """
    Print a side-by-side descriptive statistics table for multiple groups.

    Parameters
    ----------
    *groups : two or more numeric lists
    names   : optional list of group labels

    Returns
    -------
    str : formatted comparison table
    """
    if names is None:
        names = [f"Group {i + 1}" for i in range(len(groups))]
    if len(names) != len(groups):
        raise ValueError("len(names) must equal number of groups")

    stats_list = [describe(g, name=names[i]) for i, g in enumerate(groups)]

    keys = ["n", "mean", "std", "median", "q1", "q3", "iqr", "skewness"]
    col_w = max(max(len(n) for n in names), 10) + 2
    header = f"{'Statistic':<16}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = "-" * len(header)
    rows = [header, sep]

    for key in keys:
        row = f"{key:<16}"
        for ds in stats_list:
            val = getattr(ds, key)
            row += f"{val:>{col_w}.4f}"
        rows.append(row)

    return "\n".join(rows)


__all__ = [
    "DescriptiveStats",
    "describe",
    "five_number_summary",
    "detect_outliers",
    "frequency_table",
    "compare_groups",
]
