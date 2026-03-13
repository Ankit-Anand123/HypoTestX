"""
Correlation tests — pure Python implementations.

Tests
-----
- Pearson product-moment correlation
- Spearman rank-order correlation
- Point-biserial correlation
"""

from typing import List

from ..core.exceptions import DataFormatError, InsufficientDataError
from ..core.result import HypoResult
from ..core.validators import (
    validate_alpha,
    validate_alternative,
    validate_data,
    validate_paired_data,
)
from ..math.basic import abs_value, sqrt
from ..math.distributions import StudentT
from ..math.statistics import mean, std

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rank_data(data: List[float]) -> List[float]:
    """Assign average ranks to a list (ties get mean rank)."""
    n = len(data)
    indexed = sorted(enumerate(data), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson_r(x: List[float], y: List[float]) -> float:
    """Compute Pearson r between two equal-length lists of floats."""
    len(x)
    x_mean = mean(x)
    y_mean = mean(y)

    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den_x = sum((xi - x_mean) ** 2 for xi in x)
    den_y = sum((yi - y_mean) ** 2 for yi in y)

    denom = sqrt(den_x * den_y)
    if denom == 0.0:
        return 0.0
    return num / denom


def _r_to_pvalue(r: float, n: int, alternative: str) -> tuple:
    """Convert Pearson r to a p-value using the t-distribution."""
    if abs_value(r) >= 1.0:
        return 0.0, float("inf") * (1 if r >= 0 else -1)

    df = n - 2
    t_stat = r * sqrt(df) / sqrt(1 - r**2)
    t_dist = StudentT(df)

    if alternative == "two-sided":
        p = 2 * (1 - t_dist.cdf(abs_value(t_stat)))
    elif alternative == "greater":
        p = 1 - t_dist.cdf(t_stat)
    else:  # less
        p = t_dist.cdf(t_stat)

    return max(0.0, min(1.0, p)), t_stat


# ---------------------------------------------------------------------------
# Pearson Correlation
# ---------------------------------------------------------------------------


def pearson_correlation(
    x: List[float],
    y: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypoResult:
    """
    Pearson product-moment correlation coefficient and test.

    Tests whether the linear relationship between *x* and *y* is significantly
    different from zero.  Assumes both variables are approximately continuous
    and bivariate normal.

    Args:
        x: First variable
        y: Second variable (same length as *x*)
        alpha: Significance level
        alternative: 'two-sided', 'greater' (positive correlation), or 'less'

    Returns:
        HypoResult with statistic=t, effect_size=r (Pearson r)

    Examples:
        >>> result = pearson_correlation([1,2,3,4,5], [2,4,5,4,5])
        >>> print(f"r = {result.effect_size:.3f}, p = {result.p_value:.4f}")
    """
    x, y = validate_paired_data(x, y)
    validate_alpha(alpha)
    validate_alternative(alternative)
    n = len(x)

    if n < 3:
        raise InsufficientDataError("Pearson correlation requires at least 3 data points")

    r = _pearson_r(x, y)
    p_value, t_stat = _r_to_pvalue(r, n, alternative)

    df = n - 2
    df = n - 2
    # Fisher's z transformation for confidence interval of r
    import math

    if abs_value(r) < 1.0:
        z_r = 0.5 * math.log((1 + r) / (1 - r))
        se_z = 1.0 / sqrt(n - 3) if n > 3 else float("inf")
        z_crit = 1.96 if alternative == "two-sided" else 1.645
        ci_z_low = z_r - z_crit * se_z
        ci_z_high = z_r + z_crit * se_z
        r_low = (math.exp(2 * ci_z_low) - 1) / (math.exp(2 * ci_z_low) + 1)
        r_high = (math.exp(2 * ci_z_high) - 1) / (math.exp(2 * ci_z_high) + 1)
        ci = (r_low, r_high)
    else:
        ci = (r, r)

    data_summary = {
        "n": n,
        "pearson_r": r,
        "t_statistic": t_stat,
        "df": df,
        "x_mean": mean(x),
        "y_mean": mean(y),
        "x_std": std(x),
        "y_std": std(y),
    }

    significance = "significant" if p_value < alpha else "not significant"
    interpretation = (
        f"The Pearson correlation is {significance} "
        f"(r({df}) = {r:.4f}, t = {t_stat:.4f}, p = {p_value:.4f}). "
        + (
            f"There is a {'positive' if r > 0 else 'negative'} linear relationship between the variables."  # noqa: E501
            if p_value < alpha
            else "No significant linear relationship detected."
        )
    )

    return HypoResult(
        test_name="Pearson Correlation",
        statistic=t_stat,
        p_value=p_value,
        effect_size=r,
        effect_size_name="r",
        confidence_interval=ci,
        degrees_of_freedom=df,
        sample_sizes=n,
        alpha=alpha,
        alternative=alternative,
        interpretation=interpretation,
        data_summary=data_summary,
    )


# ---------------------------------------------------------------------------
# Spearman Rank Correlation
# ---------------------------------------------------------------------------


def spearman_correlation(
    x: List[float],
    y: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypoResult:
    """
    Spearman rank-order correlation coefficient and test.

    A non-parametric measure of monotonic association.  Computed by ranking
    both variables and calculating their Pearson correlation.

    Args:
        x: First variable
        y: Second variable (same length as *x*)
        alpha: Significance level
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        HypoResult with statistic=t, effect_size=rho (Spearman rho)

    Examples:
        >>> result = spearman_correlation([3,1,4,1,5], [9,2,6,5,3])
        >>> print(f"rho = {result.effect_size:.3f}")
    """
    x, y = validate_paired_data(x, y)
    validate_alpha(alpha)
    validate_alternative(alternative)
    n = len(x)

    if n < 3:
        raise InsufficientDataError("Spearman correlation requires at least 3 data points")

    # Compute ranks
    x_ranks = _rank_data(x)
    y_ranks = _rank_data(y)

    # Spearman rho is Pearson r on the ranks
    rho = _pearson_r(x_ranks, y_ranks)
    p_value, t_stat = _r_to_pvalue(rho, n, alternative)

    df = n - 2

    data_summary = {
        "n": n,
        "spearman_rho": rho,
        "t_statistic": t_stat,
        "df": df,
    }

    significance = "significant" if p_value < alpha else "not significant"
    interpretation = (
        f"The Spearman correlation is {significance} "
        f"(rho({df}) = {rho:.4f}, t = {t_stat:.4f}, p = {p_value:.4f}). "
        + (
            f"There is a {'positive' if rho > 0 else 'negative'} monotonic relationship between the variables."  # noqa: E501
            if p_value < alpha
            else "No significant monotonic relationship detected."
        )
    )

    return HypoResult(
        test_name="Spearman Rank Correlation",
        statistic=t_stat,
        p_value=p_value,
        effect_size=rho,
        effect_size_name="r",
        confidence_interval=None,
        degrees_of_freedom=df,
        sample_sizes=n,
        alpha=alpha,
        alternative=alternative,
        interpretation=interpretation,
        data_summary=data_summary,
    )


# ---------------------------------------------------------------------------
# Point-Biserial Correlation
# ---------------------------------------------------------------------------


def point_biserial_correlation(
    continuous: List[float],
    binary: List,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypoResult:
    """
    Point-biserial correlation between a continuous and a dichotomous variable.

    Equivalent to Pearson correlation when one variable is binary (0/1 coded).
    Tests whether the mean of the continuous variable differs across the two groups.

    Args:
        continuous: Continuous-scale measurements
        binary: Binary group indicator (must contain exactly 2 unique values)
        alpha: Significance level
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        HypoResult with statistic=t, effect_size=r_pb

    Examples:
        >>> result = point_biserial_correlation(
        ...     [5.5, 6.1, 7.2, 4.8, 5.9],
        ...     [0, 1, 1, 0, 0]
        ... )
    """
    continuous = validate_data(continuous, 2, "continuous")
    validate_alpha(alpha)
    validate_alternative(alternative)

    if len(binary) != len(continuous):
        raise DataFormatError(
            f"continuous and binary must have the same length, "
            f"got {len(continuous)} and {len(binary)}"
        )

    unique_vals = list(set(binary))
    if len(unique_vals) != 2:
        raise DataFormatError(
            f"binary must contain exactly 2 unique values, got {len(unique_vals)}: {unique_vals}"
        )

    # Encode binary variable as 0/1
    lo, hi = sorted(unique_vals)
    binary_coded = [0.0 if v == lo else 1.0 for v in binary]

    n = len(continuous)
    n1 = sum(binary_coded)
    n0 = n - n1

    if n1 == 0 or n0 == 0:
        raise DataFormatError("Both groups must have at least one observation")

    group1 = [continuous[i] for i in range(n) if binary_coded[i] == 1.0]
    group0 = [continuous[i] for i in range(n) if binary_coded[i] == 0.0]

    M1 = mean(group1)
    M0 = mean(group0)
    S_y = std(continuous, ddof=0)  # population std for r_pb formula

    if S_y == 0.0:
        raise ValueError("Standard deviation of 'continuous' is zero")

    # Point-biserial r formula
    r_pb = ((M1 - M0) / S_y) * sqrt((n1 * n0) / (n * n))

    p_value, t_stat = _r_to_pvalue(r_pb, n, alternative)
    df = n - 2

    data_summary = {
        "n": n,
        "n_group0": int(n0),
        "n_group1": int(n1),
        "mean_group0": M0,
        "mean_group1": M1,
        "overall_std": S_y,
        "r_pb": r_pb,
        "t_statistic": t_stat,
        "df": df,
        "group0_label": lo,
        "group1_label": hi,
    }

    significance = "significant" if p_value < alpha else "not significant"
    interpretation = (
        f"The point-biserial correlation is {significance} "
        f"(r_pb({df}) = {r_pb:.4f}, t = {t_stat:.4f}, p = {p_value:.4f}). "
        + (
            f"The continuous variable differs significantly across the two groups "
            f"(mean_{hi} = {M1:.3f}, mean_{lo} = {M0:.3f})."
            if p_value < alpha
            else "No significant difference in the continuous variable across the two groups."
        )
    )

    return HypoResult(
        test_name="Point-Biserial Correlation",
        statistic=t_stat,
        p_value=p_value,
        effect_size=r_pb,
        effect_size_name="r",
        confidence_interval=None,
        degrees_of_freedom=df,
        sample_sizes=n,
        alpha=alpha,
        alternative=alternative,
        interpretation=interpretation,
        data_summary=data_summary,
    )
