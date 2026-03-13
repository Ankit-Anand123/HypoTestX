"""
Required-sample-size calculations.

All functions solve for the minimum n needed so that power >= target_power.
Uses bisection search over n so every formula is consistent with power/analysis.py.

Provides
--------
n_ttest_one_sample(effect_size, alpha, power, alternative)
n_ttest_two_sample(effect_size, alpha, power, alternative)
n_ttest_paired(effect_size, alpha, power, alternative)
n_anova(effect_size, k, alpha, power)
n_chi_square(effect_size, df, alpha, power)
n_correlation(r, alpha, power, alternative)
sample_size_summary(...)
"""

from typing import Optional

from ..math.basic import sqrt
from .analysis import (
    power_anova,
    power_chi_square,
    power_correlation,
    power_ttest_one_sample,
    power_ttest_paired,
    power_ttest_two_sample,
)

# ---------------------------------------------------------------------------
# Internal bisection solver
# ---------------------------------------------------------------------------


def _solve_n(
    power_fn,
    target_power: float,
    n_low: int = 2,
    n_high: int = 100_000,
    **kwargs,
) -> int:
    """
    Binary search for smallest integer n such that power_fn(n, **kwargs) >= target_power.
    Returns n_high if power never reaches target within the search range.
    """
    if target_power <= 0 or target_power >= 1:
        raise ValueError("target_power must be strictly between 0 and 1")

    # Expand upper bound if needed
    while power_fn(n_high, **kwargs) < target_power:
        n_high = n_high * 2
        if n_high > 10_000_000:
            return n_high  # give up

    lo, hi = n_low, n_high
    while lo < hi:
        mid = (lo + hi) // 2
        if power_fn(mid, **kwargs) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------------------------------------------------------------------
# One-sample t-test
# ---------------------------------------------------------------------------


def n_ttest_one_sample(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> int:
    """
    Minimum n for a one-sample t-test.

    Parameters
    ----------
    effect_size : Cohen's d  (|mu - mu0| / sigma)
    alpha       : Type I error rate
    power       : desired power (1 - beta), default 0.80
    alternative : 'two-sided', 'greater', or 'less'

    Returns
    -------
    int : minimum sample size per test
    """
    fn = lambda n: power_ttest_one_sample(
        effect_size, n, alpha=alpha, alternative=alternative
    )
    return _solve_n(fn, power)


# ---------------------------------------------------------------------------
# Two-sample t-test
# ---------------------------------------------------------------------------


def n_ttest_two_sample(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> int:
    """
    Minimum n per group (balanced) for an independent-samples t-test.

    Parameters
    ----------
    effect_size : Cohen's d
    alpha       : Type I error rate
    power       : desired power
    alternative : 'two-sided', 'greater', or 'less'

    Returns
    -------
    int : minimum n per group; total N = 2 * returned value
    """
    fn = lambda n: power_ttest_two_sample(
        effect_size, n, n, alpha=alpha, alternative=alternative
    )
    return _solve_n(fn, power)


# ---------------------------------------------------------------------------
# Paired t-test
# ---------------------------------------------------------------------------


def n_ttest_paired(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> int:
    """
    Minimum number of pairs for a paired t-test.

    Parameters
    ----------
    effect_size : Cohen's d on the differences
    alpha       : Type I error rate
    power       : desired power
    alternative : 'two-sided', 'greater', or 'less'

    Returns
    -------
    int : minimum number of pairs
    """
    fn = lambda n: power_ttest_paired(
        effect_size, n, alpha=alpha, alternative=alternative
    )
    return _solve_n(fn, power)


# ---------------------------------------------------------------------------
# One-way ANOVA
# ---------------------------------------------------------------------------


def n_anova(
    effect_size: float,
    k: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Minimum n per group (balanced) for a one-way ANOVA.

    Parameters
    ----------
    effect_size : Cohen's f  (sigma_means / sigma_within)
                  Conventions: small=0.10, medium=0.25, large=0.40
    k           : number of groups
    alpha       : Type I error rate
    power       : desired power

    Returns
    -------
    int : minimum n per group; total N = k * returned value
    """
    if k < 2:
        raise ValueError("ANOVA requires at least 2 groups")
    fn = lambda n: power_anova(effect_size, n, k, alpha=alpha)
    return _solve_n(fn, power)


# ---------------------------------------------------------------------------
# Chi-square
# ---------------------------------------------------------------------------


def n_chi_square(
    effect_size: float,
    df: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Minimum total N for a chi-square test.

    Parameters
    ----------
    effect_size : Cohen's w
                  Conventions: small=0.10, medium=0.30, large=0.50
    df          : degrees of freedom
    alpha       : Type I error rate
    power       : desired power

    Returns
    -------
    int : minimum total sample size
    """
    fn = lambda n: power_chi_square(effect_size, n, df, alpha=alpha)
    return _solve_n(fn, power)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def n_correlation(
    r: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> int:
    """
    Minimum n for a Pearson/Spearman correlation test.

    Parameters
    ----------
    r           : expected Pearson r (effect size, -1..1, 0 excluded)
    alpha       : Type I error rate
    power       : desired power
    alternative : 'two-sided', 'greater', or 'less'

    Returns
    -------
    int : minimum sample size
    """
    fn = lambda n: power_correlation(r, n, alpha=alpha, alternative=alternative)
    return _solve_n(fn, power, n_low=4)


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------


def sample_size_summary(
    test_type: str,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    **kwargs,
) -> str:
    """
    Human-readable sample-size summary.

    Parameters
    ----------
    test_type   : 'one_sample_t', 'two_sample_t', 'paired_t', 'anova',
                  'chi_square', 'correlation'
    effect_size : appropriate effect-size measure for the test
    alpha       : Type I error rate (default 0.05)
    power       : desired power (default 0.80)
    **kwargs    : extra arguments (e.g. k for ANOVA, df for chi-square)

    Returns
    -------
    str : formatted summary string
    """
    dispatch = {
        "one_sample_t": lambda: n_ttest_one_sample(
            effect_size, alpha, power, kwargs.get("alternative", "two-sided")
        ),
        "two_sample_t": lambda: n_ttest_two_sample(
            effect_size, alpha, power, kwargs.get("alternative", "two-sided")
        ),
        "paired_t": lambda: n_ttest_paired(
            effect_size, alpha, power, kwargs.get("alternative", "two-sided")
        ),
        "anova": lambda: n_anova(effect_size, kwargs.get("k", 3), alpha, power),
        "chi_square": lambda: n_chi_square(
            effect_size, kwargs.get("df", 1), alpha, power
        ),
        "correlation": lambda: n_correlation(
            effect_size, alpha, power, kwargs.get("alternative", "two-sided")
        ),
    }
    if test_type not in dispatch:
        raise ValueError(f"test_type must be one of: {list(dispatch.keys())}")

    n = dispatch[test_type]()
    labels = {
        "one_sample_t": f"n = {n} (total)",
        "paired_t": f"n = {n} pairs",
        "two_sample_t": f"n = {n} per group  (total = {2 * n})",
        "anova": f"n = {n} per group  (total = {kwargs.get('k', 3) * n})",
        "chi_square": f"n = {n} (total)",
        "correlation": f"n = {n} (total)",
    }

    lines = [
        f"Sample Size Analysis: {test_type.replace('_', ' ').title()}",
        f"  Effect size   : {effect_size:.4f}",
        f"  alpha         : {alpha}",
        f"  Target power  : {power}",
        f"  Required size : {labels[test_type]}",
    ]
    return "\n".join(lines)


__all__ = [
    "n_ttest_one_sample",
    "n_ttest_two_sample",
    "n_ttest_paired",
    "n_anova",
    "n_chi_square",
    "n_correlation",
    "sample_size_summary",
]
