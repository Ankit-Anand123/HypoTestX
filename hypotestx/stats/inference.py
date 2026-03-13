"""
Frequentist inference helpers: confidence intervals for means and proportions.

Provides
--------
confidence_interval_mean(data, confidence, alternative)
    -> (lower, upper)  Student-t based CI for the population mean.

confidence_interval_proportion(successes, n, confidence, method)
    -> (lower, upper)  Wilson score or normal approximation CI.

confidence_interval_difference_of_means(group1, group2, confidence)
    -> (lower, upper)  Welch / unpooled CI for the difference mu1 - mu2.

z_test_one_sample(data, mu0, alpha, alternative)
    -> (z_stat, p_value)  Large-sample z-test.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from ..math.basic import sqrt
from ..math.distributions import Normal, StudentT
from ..math.statistics import mean, std, variance

# ---------------------------------------------------------------------------
# CI for a single mean  (t-based)
# ---------------------------------------------------------------------------


def confidence_interval_mean(
    data: Sequence[float],
    confidence: float = 0.95,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    t-based confidence interval for the population mean.

    Parameters
    ----------
    data        : numeric sequence
    confidence  : confidence level, e.g. 0.95
    alternative : 'two-sided' (default), 'greater', or 'less'

    Returns
    -------
    (lower, upper) : confidence interval bounds
    """
    data = [float(x) for x in data]
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points to compute a CI")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")

    xbar = mean(data)
    se = std(data) / sqrt(n)
    df = n - 1
    t = StudentT(df)
    alpha = 1.0 - confidence

    if alternative == "two-sided":
        t_crit = t.ppf(1.0 - alpha / 2.0)
        return xbar - t_crit * se, xbar + t_crit * se
    elif alternative == "greater":
        t_crit = t.ppf(1.0 - alpha)
        return xbar - t_crit * se, float("inf")
    elif alternative == "less":
        t_crit = t.ppf(1.0 - alpha)
        return float("-inf"), xbar + t_crit * se
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")


# ---------------------------------------------------------------------------
# CI for a proportion
# ---------------------------------------------------------------------------


def confidence_interval_proportion(
    successes: int,
    n: int,
    confidence: float = 0.95,
    method: str = "wilson",
) -> Tuple[float, float]:
    """
    Confidence interval for a population proportion.

    Parameters
    ----------
    successes  : number of successes (0 <= successes <= n)
    n          : total observations
    confidence : confidence level
    method     : 'wilson' (default) or 'normal'

    Returns
    -------
    (lower, upper)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")

    p_hat = successes / n
    alpha = 1.0 - confidence
    norm = Normal(0, 1)
    z = norm.ppf(1.0 - alpha / 2.0)

    if method == "normal":
        se = sqrt(p_hat * (1 - p_hat) / n)
        lower = max(0.0, p_hat - z * se)
        upper = min(1.0, p_hat + z * se)

    elif method == "wilson":
        z2 = z * z
        denom = 1 + z2 / n
        centre = (p_hat + z2 / (2 * n)) / denom
        half = (z / denom) * sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
        lower = max(0.0, centre - half)
        upper = min(1.0, centre + half)

    else:
        raise ValueError("method must be 'wilson' or 'normal'")

    return lower, upper


# ---------------------------------------------------------------------------
# CI for the difference between two means (Welch)
# ---------------------------------------------------------------------------


def confidence_interval_difference_of_means(
    group1: Sequence[float],
    group2: Sequence[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Welch unpooled confidence interval for (mu1 - mu2).

    Returns
    -------
    (lower, upper)
    """
    g1 = [float(x) for x in group1]
    g2 = [float(x) for x in group2]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Both groups need at least 2 observations")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")

    m1, m2 = mean(g1), mean(g2)
    s1, s2 = std(g1), std(g2)
    var1, var2 = s1**2 / n1, s2**2 / n2
    se = sqrt(var1 + var2)
    diff = m1 - m2

    # Welch-Satterthwaite df
    df = (var1 + var2) ** 2 / (var1**2 / (n1 - 1) + var2**2 / (n2 - 1))

    t = StudentT(max(1, int(df)))
    alpha = 1.0 - confidence
    t_crit = t.ppf(1.0 - alpha / 2.0)
    return diff - t_crit * se, diff + t_crit * se


# ---------------------------------------------------------------------------
# One-sample z-test  (large-sample)
# ---------------------------------------------------------------------------


def z_test_one_sample(
    data: Sequence[float],
    mu0: float,
    sigma: Optional[float] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    One-sample z-test for the population mean.

    Parameters
    ----------
    data        : numeric sequence
    mu0         : null hypothesis mean
    sigma       : known population std (if None, sample std is used)
    alpha       : significance level
    alternative : 'two-sided', 'greater', or 'less'

    Returns
    -------
    (z_statistic, p_value)
    """
    data = [float(x) for x in data]
    n = len(data)
    if n < 1:
        raise ValueError("Data must be non-empty")

    xbar = mean(data)
    s = sigma if sigma is not None else std(data)
    se = s / sqrt(n)

    if se == 0:
        raise ValueError("Standard error is zero; all values are identical")

    z_stat = (xbar - mu0) / se
    norm = Normal(0, 1)

    if alternative == "two-sided":
        p = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    elif alternative == "greater":
        p = 1.0 - norm.cdf(z_stat)
    elif alternative == "less":
        p = norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return z_stat, min(max(p, 0.0), 1.0)


__all__ = [
    "confidence_interval_mean",
    "confidence_interval_proportion",
    "confidence_interval_difference_of_means",
    "z_test_one_sample",
]
