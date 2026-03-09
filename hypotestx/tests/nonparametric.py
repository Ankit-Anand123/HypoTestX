"""
Non-parametric statistical tests — pure Python implementations.

Tests
-----
- Mann-Whitney U  (two independent samples)
- Wilcoxon Signed-Rank  (one sample / paired samples)
- Kruskal-Wallis H  (k independent samples)
"""
from typing import List, Optional, Tuple

from ..math.distributions import Normal, ChiSquare
from ..math.basic import sqrt, abs_value
from ..math.statistics import mean, std
from ..core.result import HypoResult
from ..core.validators import validate_data, validate_alpha, validate_alternative, validate_groups


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_data(data: List[float]) -> List[float]:
    """
    Assign ranks to a flat list of values, using average ranks for ties.

    Returns a list of ranks in the same original order as *data*.
    """
    n = len(data)
    indexed = sorted(enumerate(data), key=lambda x: x[1])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0          # 1-based average rank
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks


def _tie_correction(data: List[float]) -> float:
    """
    Compute the sum of (t^3 - t) for each group of tied values.
    Used to correct variance in non-parametric tests.
    """
    from collections import Counter
    counts = Counter(data)
    return sum(t ** 3 - t for t in counts.values() if t > 1)


# ---------------------------------------------------------------------------
# Mann-Whitney U Test
# ---------------------------------------------------------------------------

def mann_whitney_u(
    group1: List[float],
    group2: List[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypoResult:
    """
    Mann-Whitney U test for two independent samples.

    A non-parametric alternative to the two-sample t-test that tests whether
    one group tends to have larger values than the other.

    Uses a normal approximation (with tie correction) for the p-value.

    Args:
        group1: First group sample values
        group2: Second group sample values
        alpha: Significance level (default 0.05)
        alternative: 'two-sided', 'greater' (group1 > group2), or 'less'

    Returns:
        HypoResult with statistic=U1, effect_size=rank-biserial r

    Examples:
        >>> result = mann_whitney_u([1, 2, 3, 4], [5, 6, 7, 8])
        >>> print(result.summary())
    """
    group1 = validate_data(group1, 2, "group1")
    group2 = validate_data(group2, 2, "group2")
    validate_alpha(alpha)
    validate_alternative(alternative)

    n1, n2 = len(group1), len(group2)
    N = n1 + n2

    # Combine groups and rank
    combined_vals = group1 + group2
    combined_groups = [1] * n1 + [2] * n2
    all_ranks = _rank_data(combined_vals)

    # Sum of ranks for group 1
    R1 = sum(rank for rank, grp in zip(all_ranks, combined_groups) if grp == 1)

    # U statistics
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1

    # Mean under H0
    mean_U = n1 * n2 / 2.0

    # Variance with tie correction
    tie_sum = _tie_correction(combined_vals)
    var_U = (n1 * n2 / 12.0) * (N + 1 - tie_sum / (N * (N - 1))) if N > 1 else 0
    if var_U <= 0:
        var_U = n1 * n2 * (N + 1) / 12.0   # fallback without tie correction

    std_U = sqrt(var_U)

    # Z statistic: U1 measures tendency of group1 to exceed group2
    Z = (U1 - mean_U) / std_U

    # p-value
    norm = Normal()
    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs_value(Z)))
    elif alternative == "greater":
        p_value = 1 - norm.cdf(Z)
    else:  # less
        p_value = norm.cdf(Z)

    # Continuity correction clamp
    p_value = max(0.0, min(1.0, p_value))

    # Effect size: rank-biserial correlation  r = (U1 - U2) / (n1 * n2)
    effect_size = (U1 - U2) / (n1 * n2)

    R2 = sum(all_ranks) - R1
    data_summary = {
        "n1": n1,
        "n2": n2,
        "rank_sum_group1": R1,
        "rank_sum_group2": R2,
        "mean_rank_group1": R1 / n1,
        "mean_rank_group2": R2 / n2,
        "U1": U1,
        "U2": U2,
        "z_statistic": Z,
    }

    significance = "significant" if p_value < alpha else "not significant"
    interpretation = (
        f"The Mann-Whitney U test is {significance} "
        f"(U = {U1:.1f}, Z = {Z:.4f}, p = {p_value:.4f}). "
        + (
            "Group 1 values tend to be larger than Group 2."
            if U1 > mean_U and p_value < alpha
            else "No significant directional difference detected."
        )
    )

    return HypoResult(
        test_name="Mann-Whitney U Test",
        statistic=U1,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_name="rank-biserial r",
        confidence_interval=None,
        degrees_of_freedom=None,
        sample_sizes=(n1, n2),
        alpha=alpha,
        alternative=alternative,
        interpretation=interpretation,
        data_summary=data_summary,
    )


# ---------------------------------------------------------------------------
# Wilcoxon Signed-Rank Test
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(
    x: List[float],
    y: Optional[List[float]] = None,
    mu: float = 0.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypoResult:
    """
    Wilcoxon signed-rank test.

    Tests whether the median of differences (or of a single sample relative
    to *mu*) differs from zero.  Non-parametric alternative to the one-sample
    or paired t-test.

    Args:
        x: Data values (or differences if *y* is not supplied)
        y: Optional second paired sample; if provided, differences = x - y
        mu: Hypothesised median under H0 (default 0)
        alpha: Significance level
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        HypoResult with statistic=W+ (sum of positive ranks)

    Examples:
        >>> result = wilcoxon_signed_rank([1, 2, 3, 4, 5], mu=2)
        >>> result = wilcoxon_signed_rank(before, after)   # paired
    """
    x = validate_data(x, 2, "x")
    validate_alpha(alpha)
    validate_alternative(alternative)

    if y is not None:
        y = validate_data(y, 2, "y")
        if len(x) != len(y):
            from ..core.exceptions import DataFormatError
            raise DataFormatError(
                f"x and y must have the same length for a paired test, "
                f"got {len(x)} and {len(y)}"
            )
        differences = [xi - yi for xi, yi in zip(x, y)]
    else:
        differences = [xi - mu for xi in x]

    # Drop zero differences
    nonzero = [d for d in differences if d != 0.0]
    n = len(nonzero)

    if n == 0:
        raise ValueError("All differences are zero — test cannot be performed")

    # Rank absolute differences
    abs_diffs = [abs_value(d) for d in nonzero]
    ranks = _rank_data(abs_diffs)

    # W+ = sum of ranks for positive differences
    W_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    W_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)

    # Expected value and variance under H0
    expected_W = n * (n + 1) / 4.0
    tie_sum = _tie_correction(abs_diffs)
    var_W = (n * (n + 1) * (2 * n + 1) / 24.0) - (tie_sum / 48.0)
    if var_W <= 0:
        var_W = n * (n + 1) * (2 * n + 1) / 24.0

    std_W = sqrt(var_W)

    # Z statistic (using W+)
    Z = (W_plus - expected_W) / std_W

    # p-value
    norm = Normal()
    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs_value(Z)))
    elif alternative == "greater":
        p_value = 1 - norm.cdf(Z)
    else:  # less
        p_value = norm.cdf(Z)

    p_value = max(0.0, min(1.0, p_value))

    # Effect size: r = Z / sqrt(n)
    effect_size = Z / sqrt(n)

    n_zeros = len(differences) - n
    data_summary = {
        "n_pairs": len(differences),
        "n_nonzero": n,
        "n_zero_diffs": n_zeros,
        "W_plus": W_plus,
        "W_minus": W_minus,
        "expected_W": expected_W,
        "z_statistic": Z,
    }

    significance = "significant" if p_value < alpha else "not significant"
    interpretation = (
        f"The Wilcoxon signed-rank test is {significance} "
        f"(W+ = {W_plus:.1f}, Z = {Z:.4f}, p = {p_value:.4f})."
    )

    return HypoResult(
        test_name="Wilcoxon Signed-Rank Test",
        statistic=W_plus,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_name="rank-biserial r",
        confidence_interval=None,
        degrees_of_freedom=None,
        sample_sizes=n,
        alpha=alpha,
        alternative=alternative,
        interpretation=interpretation,
        data_summary=data_summary,
    )


# ---------------------------------------------------------------------------
# Kruskal-Wallis H Test
# ---------------------------------------------------------------------------

def kruskal_wallis(
    *groups: List[float],
    alpha: float = 0.05,
) -> HypoResult:
    """
    Kruskal-Wallis H test for k independent groups.

    Non-parametric one-way ANOVA. Tests whether the population medians of all
    groups are equal.  P-value is obtained from the chi-square distribution
    with k-1 degrees of freedom.

    Args:
        *groups: Two or more group samples (each a list of floats)
        alpha: Significance level

    Returns:
        HypoResult with statistic=H, effect_size=eta-squared

    Examples:
        >>> result = kruskal_wallis([1,2,3], [4,5,6], [7,8,9])
        >>> print(result.summary())
    """
    groups = validate_groups(*groups, min_size=2, min_groups=2)
    validate_alpha(alpha)

    k = len(groups)
    group_sizes = [len(g) for g in groups]
    N = sum(group_sizes)

    # Combine all observations and rank
    combined_vals: List[float] = []
    group_labels: List[int] = []
    for idx, g in enumerate(groups):
        combined_vals.extend(g)
        group_labels.extend([idx] * len(g))

    all_ranks = _rank_data(combined_vals)

    # Sum of ranks per group
    rank_sums = [0.0] * k
    for rank, label in zip(all_ranks, group_labels):
        rank_sums[label] += rank

    # H statistic
    h_inner = sum(rank_sums[i] ** 2 / group_sizes[i] for i in range(k))
    H = (12.0 / (N * (N + 1))) * h_inner - 3 * (N + 1)

    # Tie correction
    tie_sum = _tie_correction(combined_vals)
    tie_factor = 1.0 - tie_sum / (N ** 3 - N) if N > 1 else 1.0
    if tie_factor > 0:
        H = H / tie_factor

    # p-value from chi-square with k-1 df
    df = k - 1
    chi2_dist = ChiSquare(df)
    p_value = 1 - chi2_dist.cdf(H) if H >= 0 else 1.0
    p_value = max(0.0, min(1.0, p_value))

    # Effect size: eta-squared = (H - k + 1) / (N - k)
    if N > k:
        eta_sq = (H - k + 1) / (N - k)
        eta_sq = max(0.0, min(1.0, eta_sq))
    else:
        eta_sq = None

    mean_ranks = [rank_sums[i] / group_sizes[i] for i in range(k)]
    data_summary = {
        "n_groups": k,
        "group_sizes": group_sizes,
        "rank_sums": rank_sums,
        "mean_ranks": mean_ranks,
        "N_total": N,
        "H_uncorrected": (12.0 / (N * (N + 1))) * h_inner - 3 * (N + 1),
    }

    significance = "significant" if p_value < alpha else "not significant"
    interpretation = (
        f"The Kruskal-Wallis test is {significance} "
        f"(H({df}) = {H:.4f}, p = {p_value:.4f}). "
        + (
            "At least one group differs significantly from the others."
            if p_value < alpha
            else "No significant difference in central tendency across groups."
        )
    )

    return HypoResult(
        test_name="Kruskal-Wallis H Test",
        statistic=H,
        p_value=p_value,
        effect_size=eta_sq,
        effect_size_name="eta-squared",
        confidence_interval=None,
        degrees_of_freedom=df,
        sample_sizes=tuple(group_sizes),
        alpha=alpha,
        alternative="two-sided",
        interpretation=interpretation,
        data_summary=data_summary,
    )
