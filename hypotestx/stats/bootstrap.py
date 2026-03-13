"""
Bootstrap resampling for confidence intervals and hypothesis tests.

Uses only Python's built-in `random` module for resampling — no numpy.

Provides
--------
bootstrap_ci(data, statistic_fn, n_resamples, ci, method, seed)
    -> (lower, upper, bootstrap_distribution)

bootstrap_two_sample_ci(group1, group2, diff_fn, n_resamples, ci, seed)
    -> (lower, upper, bootstrap_distribution)

bootstrap_mean_ci(data, ci, n_resamples, seed)
    -> (lower, upper)

bootstrap_test(data, statistic_fn, null_value, n_resamples, alternative, seed)
    -> (p_value, observed_stat, bootstrap_distribution)

permutation_test(group1, group2, statistic_fn, n_resamples, alternative, seed)
    -> (p_value, observed_stat, permutation_distribution)
"""

import random
from typing import Callable, List, Optional, Tuple

from ..math.statistics import mean, percentile

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bootstrap_sample(data: List[float], rng: random.Random) -> List[float]:
    """Draw a bootstrap resample of the same length with replacement."""
    n = len(data)
    return [data[rng.randrange(n)] for _ in range(n)]


def _percentile_ci(
    boot_stats: List[float],
    ci: float,
) -> Tuple[float, float]:
    """Percentile confidence interval from bootstrap distribution."""
    alpha_half = (1.0 - ci) / 2.0
    lower = percentile(boot_stats, alpha_half * 100.0)
    upper = percentile(boot_stats, (1.0 - alpha_half) * 100.0)
    return lower, upper


def _bca_ci(
    data: List[float],
    statistic_fn: Callable,
    boot_stats: List[float],
    observed: float,
    ci: float,
) -> Tuple[float, float]:
    """
    Bias-corrected and accelerated (BCa) confidence interval.

    Uses the acceleration (jackknife) and bias-correction factors.
    """
    import math

    from ..math.basic import ln
    from ..math.distributions import Normal

    n = len(data)
    n_boot = len(boot_stats)

    # --- Bias correction z0 ---
    count_below = sum(1 for s in boot_stats if s < observed)
    p0 = count_below / n_boot if n_boot > 0 else 0.5
    p0 = max(min(p0, 1.0 - 1e-9), 1e-9)
    norm = Normal(0, 1)
    z0 = norm.ppf(p0)

    # --- Acceleration a (jackknife) ---
    jack_stats = []
    for i in range(n):
        jack_data = data[:i] + data[i + 1 :]
        jack_stats.append(statistic_fn(jack_data))
    jack_mean = mean(jack_stats)
    diffs = [jack_mean - js for js in jack_stats]
    numer = sum(d**3 for d in diffs)
    denom = 6.0 * (sum(d**2 for d in diffs) ** 1.5)
    a = numer / denom if denom != 0 else 0.0

    # --- Adjusted quantiles ---
    alpha_half = (1.0 - ci) / 2.0
    z_alpha = norm.ppf(alpha_half)
    z_1alpha = norm.ppf(1.0 - alpha_half)

    def _adj_quantile(z_a: float) -> float:
        denom2 = 1.0 - a * (z0 + z_a)
        if denom2 == 0:
            denom2 = 1e-9
        z_adj = z0 + (z0 + z_a) / denom2
        return norm.cdf(z_adj)

    p_lo = _adj_quantile(z_alpha)
    p_hi = _adj_quantile(z_1alpha)

    lower = percentile(sorted(boot_stats), p_lo * 100.0)
    upper = percentile(sorted(boot_stats), p_hi * 100.0)
    return lower, upper


# ---------------------------------------------------------------------------
# Bootstrap CI — single sample
# ---------------------------------------------------------------------------


def bootstrap_ci(
    data: List[float],
    statistic_fn: Callable[[List[float]], float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    method: str = "percentile",
    seed: Optional[int] = None,
) -> Tuple[float, float, List[float]]:
    """
    Bootstrap confidence interval for any sample statistic.

    Parameters
    ----------
    data          : numeric list
    statistic_fn  : function(sample) -> float, e.g. ``mean`` or ``median``
    n_resamples   : number of bootstrap resamples (default 2 000)
    ci            : confidence level (default 0.95)
    method        : 'percentile' (default) or 'bca'
    seed          : random seed for reproducibility

    Returns
    -------
    (lower, upper, boot_distribution)
        lower / upper : confidence interval bounds
        boot_distribution : list of bootstrap statistic values
    """
    data = [float(x) for x in data]
    if len(data) < 2:
        raise ValueError("bootstrap_ci requires at least 2 data points")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    rng = random.Random(seed)
    observed = statistic_fn(data)
    boot_stats = [
        statistic_fn(_bootstrap_sample(data, rng)) for _ in range(n_resamples)
    ]

    if method == "percentile":
        lower, upper = _percentile_ci(boot_stats, ci)
    elif method == "bca":
        lower, upper = _bca_ci(data, statistic_fn, boot_stats, observed, ci)
    else:
        raise ValueError("method must be 'percentile' or 'bca'")

    return lower, upper, boot_stats


# ---------------------------------------------------------------------------
# Bootstrap CI — two-sample difference
# ---------------------------------------------------------------------------


def bootstrap_two_sample_ci(
    group1: List[float],
    group2: List[float],
    diff_fn: Optional[Callable[[List[float], List[float]], float]] = None,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, List[float]]:
    """
    Bootstrap CI for the difference between two independent groups.

    Parameters
    ----------
    group1, group2 : numeric lists
    diff_fn        : function(g1, g2) -> float; default = mean(g1) - mean(g2)
    n_resamples    : number of resamples
    ci             : confidence level
    seed           : random seed

    Returns
    -------
    (lower, upper, boot_distribution)
    """
    group1 = [float(x) for x in group1]
    group2 = [float(x) for x in group2]
    if diff_fn is None:
        diff_fn = lambda a, b: mean(a) - mean(b)

    rng = random.Random(seed)
    n1, n2 = len(group1), len(group2)
    boot_stats = []

    for _ in range(n_resamples):
        s1 = [group1[rng.randrange(n1)] for _ in range(n1)]
        s2 = [group2[rng.randrange(n2)] for _ in range(n2)]
        boot_stats.append(diff_fn(s1, s2))

    lower, upper = _percentile_ci(boot_stats, ci)
    return lower, upper, boot_stats


# ---------------------------------------------------------------------------
# Convenience: bootstrap mean CI
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    data: List[float],
    ci: float = 0.95,
    n_resamples: int = 2000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Bootstrap percentile CI for the mean.

    Returns
    -------
    (lower, upper)
    """
    lower, upper, _ = bootstrap_ci(
        data, mean, n_resamples=n_resamples, ci=ci, seed=seed
    )
    return lower, upper


# ---------------------------------------------------------------------------
# Bootstrap hypothesis test (one-sample)
# ---------------------------------------------------------------------------


def bootstrap_test(
    data: List[float],
    statistic_fn: Callable[[List[float]], float],
    null_value: float = 0.0,
    n_resamples: int = 2000,
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> Tuple[float, float, List[float]]:
    """
    Bootstrap hypothesis test: Is statistic_fn(data) != null_value?

    Shifts the data so that the statistic equals null_value under H0,
    then measures how extreme the observed statistic is.

    Parameters
    ----------
    data          : numeric list
    statistic_fn  : function(sample) -> float
    null_value    : value to test against (default 0)
    n_resamples   : number of resamples
    alternative   : 'two-sided', 'greater', or 'less'
    seed          : random seed

    Returns
    -------
    (p_value, observed_stat, boot_distribution)
    """
    data = [float(x) for x in data]
    observed = statistic_fn(data)
    # Shift data so null is true
    shift = null_value - observed
    h0_data = [x + shift for x in data]

    rng = random.Random(seed)
    boot_stats = [
        statistic_fn(_bootstrap_sample(h0_data, rng)) for _ in range(n_resamples)
    ]

    if alternative == "two-sided":
        p_value = sum(
            1 for s in boot_stats if abs(s - null_value) >= abs(observed - null_value)
        )
    elif alternative == "greater":
        p_value = sum(1 for s in boot_stats if s >= observed)
    elif alternative == "less":
        p_value = sum(1 for s in boot_stats if s <= observed)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    p_value = (p_value + 1) / (n_resamples + 1)  # +1 smoothing
    return p_value, observed, boot_stats


# ---------------------------------------------------------------------------
# Permutation test (two-sample)
# ---------------------------------------------------------------------------


def permutation_test(
    group1: List[float],
    group2: List[float],
    statistic_fn: Optional[Callable[[List[float], List[float]], float]] = None,
    n_resamples: int = 2000,
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> Tuple[float, float, List[float]]:
    """
    Permutation test for the difference between two independent groups.

    Null hypothesis: the two groups come from the same distribution.

    Parameters
    ----------
    group1, group2  : numeric lists
    statistic_fn    : function(g1, g2) -> float; default = mean(g1) - mean(g2)
    n_resamples     : number of permutations
    alternative     : 'two-sided', 'greater', or 'less'
    seed            : random seed

    Returns
    -------
    (p_value, observed_stat, permutation_distribution)
    """
    group1 = [float(x) for x in group1]
    group2 = [float(x) for x in group2]
    if statistic_fn is None:
        statistic_fn = lambda a, b: mean(a) - mean(b)

    n1 = len(group1)
    combined = group1 + group2
    n_total = len(combined)
    observed = statistic_fn(group1, group2)

    rng = random.Random(seed)
    perm_stats = []

    for _ in range(n_resamples):
        perm = combined[:]
        rng.shuffle(perm)
        perm_stats.append(statistic_fn(perm[:n1], perm[n1:]))

    if alternative == "two-sided":
        p_value = sum(1 for s in perm_stats if abs(s) >= abs(observed))
    elif alternative == "greater":
        p_value = sum(1 for s in perm_stats if s >= observed)
    elif alternative == "less":
        p_value = sum(1 for s in perm_stats if s <= observed)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    p_value = (p_value + 1) / (n_resamples + 1)
    return p_value, observed, perm_stats


__all__ = [
    "bootstrap_ci",
    "bootstrap_two_sample_ci",
    "bootstrap_mean_ci",
    "bootstrap_test",
    "permutation_test",
]
