"""
Categorical statistical tests — pure Python implementations.

Tests
-----
- Chi-Square Test of Independence / Goodness-of-Fit
- Fisher's Exact Test  (2×2 tables)
"""

from typing import List, Optional, Union

from ..core.exceptions import DataFormatError, InsufficientDataError
from ..core.result import HypoResult
from ..core.validators import (
    validate_alpha,
    validate_alternative,
    validate_contingency_table,
)
from ..math.basic import ln, sqrt
from ..math.distributions import ChiSquare, Normal
from ..math.special import gamma

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_expected(observed: List[List[float]]) -> List[List[float]]:
    """Compute expected cell frequencies from row/column marginals."""
    nrows = len(observed)
    ncols = len(observed[0])
    row_sums = [sum(observed[r]) for r in range(nrows)]
    col_sums = [sum(observed[r][c] for r in range(nrows)) for c in range(ncols)]
    total = sum(row_sums)
    if total == 0:
        raise DataFormatError(
            "Contingency table is all zeros — cannot compute expected frequencies"
        )
    return [
        [row_sums[r] * col_sums[c] / total for c in range(ncols)] for r in range(nrows)
    ]


def _hypergeom_pmf(k: int, N: int, K: int, n: int) -> float:
    """
    Hypergeometric PMF: P(X = k)

    Drawing *n* items from a population of *N* where *K* are successes.
    Uses log-space arithmetic to avoid overflow with large factorials.
    """
    from ..math.special import gamma  # log-gamma for numerics

    def log_binom(a: int, b: int) -> float:
        """log C(a, b) using log-gamma"""
        if b < 0 or b > a:
            return float("-inf")
        if b == 0 or b == a:
            return 0.0
        # log C(a,b) = lgamma(a+1) - lgamma(b+1) - lgamma(a-b+1)
        import math

        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    import math

    log_p = log_binom(K, k) + log_binom(N - K, n - k) - log_binom(N, n)
    if log_p == float("-inf"):
        return 0.0
    return math.exp(log_p)


# ---------------------------------------------------------------------------
# Chi-Square Test
# ---------------------------------------------------------------------------


def chi_square_test(
    observed: Union[List[List[float]], List[float]],
    expected: Optional[List[float]] = None,
    alpha: float = 0.05,
    correction: bool = False,
) -> HypoResult:
    """
    Chi-square test of independence (2-D table) or goodness-of-fit (1-D).

    For a 2-D contingency table the test checks whether row and column
    variables are independent.  For a 1-D array of observed counts it checks
    whether they follow the specified (or uniform) expected distribution.

    Args:
        observed: 2-D contingency table  *or*  1-D list of observed counts
        expected: Expected counts for each category (1-D case only).
                  If None, a uniform expected distribution is assumed.
        alpha: Significance level
        correction: Apply Yates' continuity correction (2×2 tables only)

    Returns:
        HypoResult with statistic=chi2, effect_size=Cramer's V (or phi for 2×2)

    Examples:
        >>> # 2-D contingency table
        >>> table = [[30, 10], [20, 40]]
        >>> result = chi_square_test(table)

        >>> # Goodness-of-fit
        >>> result = chi_square_test([50, 30, 20], expected=[40, 30, 30])
    """
    validate_alpha(alpha)

    # Determine if 1-D or 2-D input
    is_1d = False
    if observed and not isinstance(observed[0], (list, tuple)):
        is_1d = True

    if is_1d:
        # ── Goodness-of-fit ────────────────────────────────────────────────
        observed_1d = [float(v) for v in observed]
        k = len(observed_1d)
        if k < 2:
            raise DataFormatError("Goodness-of-fit test requires at least 2 categories")

        total = sum(observed_1d)
        if total <= 0:
            raise DataFormatError("Total observed count must be positive")

        if expected is None:
            expected_1d = [total / k] * k
        else:
            if len(expected) != k:
                raise DataFormatError(
                    f"expected must have the same length as observed ({k}), got {len(expected)}"
                )
            expected_1d = [float(v) for v in expected]
            exp_total = sum(expected_1d)
            if abs(exp_total - total) > 1e-6:
                # re-scale expected to match observed total
                expected_1d = [e * total / exp_total for e in expected_1d]

        # Check expected frequencies
        if any(e < 5 for e in expected_1d):
            import warnings

            warnings.warn(
                "Some expected frequencies are less than 5; chi-square approximation may be inaccurate."
            )

        chi2 = sum((o - e) ** 2 / e for o, e in zip(observed_1d, expected_1d) if e > 0)
        df = k - 1
        n = total
        nrows, ncols = 1, k  # for effect size formula
        table_2d = [observed_1d]

        # Effect size: w = sqrt(chi2 / n)
        effect_size = sqrt(chi2 / n) if n > 0 else None
        effect_name = "Cohen's w"

    else:
        # ── Test of independence ───────────────────────────────────────────
        table_2d = validate_contingency_table(observed)
        nrows = len(table_2d)
        ncols = len(table_2d[0])

        expected_2d = _build_expected(table_2d)
        n = sum(table_2d[r][c] for r in range(nrows) for c in range(ncols))

        # Check expected frequencies
        small_cells = sum(
            1 for r in range(nrows) for c in range(ncols) if expected_2d[r][c] < 5
        )
        if small_cells > 0:
            import warnings

            warnings.warn(
                f"{small_cells} cell(s) have expected frequency < 5; "
                "consider Fisher's exact test for 2×2 tables."
            )

        # Yates' continuity correction (2×2 only)
        if correction and nrows == 2 and ncols == 2:
            chi2 = sum(
                (abs(table_2d[r][c] - expected_2d[r][c]) - 0.5) ** 2 / expected_2d[r][c]
                for r in range(nrows)
                for c in range(ncols)
                if expected_2d[r][c] > 0
            )
        else:
            chi2 = sum(
                (table_2d[r][c] - expected_2d[r][c]) ** 2 / expected_2d[r][c]
                for r in range(nrows)
                for c in range(ncols)
                if expected_2d[r][c] > 0
            )

        df = (nrows - 1) * (ncols - 1)

        # Effect size: Cramer's V  (phi for 2×2)
        min_dim = min(nrows - 1, ncols - 1)
        if n > 0 and min_dim > 0:
            cramers_v = sqrt(chi2 / (n * min_dim))
            cramers_v = min(cramers_v, 1.0)
        else:
            cramers_v = None

        effect_size = cramers_v
        effect_name = "phi" if (nrows == 2 and ncols == 2) else "Cramer's V"

    # p-value from chi-square distribution
    chi2_dist = ChiSquare(df)
    p_value = max(0.0, min(1.0, 1 - chi2_dist.cdf(chi2)))

    data_summary = {
        "observed": table_2d if not is_1d else observed_1d,
        "expected": expected_2d if not is_1d else expected_1d,
        "n_total": n,
        "n_rows": nrows,
        "n_cols": ncols,
    }

    significance = "significant" if p_value < alpha else "not significant"
    if is_1d:
        interpretation = (
            f"The chi-square goodness-of-fit test is {significance} "
            f"(chi2({df}) = {chi2:.4f}, p = {p_value:.4f}). "
            + (
                "The observed distribution differs significantly from expected."
                if p_value < alpha
                else "No significant deviation from expected distribution."
            )
        )
    else:
        interpretation = (
            f"The chi-square test of independence is {significance} "
            f"(chi2({df}) = {chi2:.4f}, p = {p_value:.4f}). "
            + (
                "The row and column variables are significantly associated."
                if p_value < alpha
                else "No significant association found between the variables."
            )
        )

    return HypoResult(
        test_name="Chi-Square Test"
        + (" of Independence" if not is_1d else " (Goodness-of-Fit)"),
        statistic=chi2,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_name=effect_name,
        confidence_interval=None,
        degrees_of_freedom=df,
        sample_sizes=int(n),
        alpha=alpha,
        alternative="two-sided",
        interpretation=interpretation,
        data_summary=data_summary,
    )


# ---------------------------------------------------------------------------
# Fisher's Exact Test
# ---------------------------------------------------------------------------


def fisher_exact_test(
    table: List[List[float]],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypoResult:
    """
    Fisher's Exact Test for a 2×2 contingency table.

    Computes the exact p-value using the hypergeometric distribution.
    Preferred over the chi-square test when any expected cell count is < 5.

    Args:
        table: 2×2 contingency table  [[a, b], [c, d]]
        alpha: Significance level
        alternative: 'two-sided', 'greater' (more association than expected),
                     or 'less'

    Returns:
        HypoResult with statistic=odds_ratio

    Examples:
        >>> table = [[8, 2], [1, 5]]
        >>> result = fisher_exact_test(table)
        >>> print(result.p_value)
    """
    table = validate_contingency_table(table)
    validate_alpha(alpha)
    validate_alternative(alternative)

    if len(table) != 2 or len(table[0]) != 2:
        raise DataFormatError(
            "Fisher's Exact Test requires a 2×2 contingency table, "
            f"got {len(table)}×{len(table[0])}"
        )

    a, b = int(table[0][0]), int(table[0][1])
    c, d = int(table[1][0]), int(table[1][1])

    R1 = a + b  # row 1 total
    R2 = c + d  # row 2 total
    C1 = a + c  # col 1 total
    C2 = b + d  # col 2 total
    N = R1 + R2

    if N == 0:
        raise DataFormatError("Contingency table total count is zero")

    # Possible values for cell (0,0) given fixed marginals
    a_min = max(0, R1 - C2)
    a_max = min(R1, C1)

    # Compute probability for each possible table
    pmf = {}
    for k in range(a_min, a_max + 1):
        pmf[k] = _hypergeom_pmf(k, N, C1, R1)

    p_observed = pmf.get(a, 0.0)

    if alternative == "two-sided":
        p_value = sum(p for p in pmf.values() if p <= p_observed + 1e-10)
    elif alternative == "greater":
        p_value = sum(p for k, p in pmf.items() if k >= a)
    else:  # less
        p_value = sum(p for k, p in pmf.items() if k <= a)

    p_value = max(0.0, min(1.0, p_value))

    # Odds ratio (with Haldane-Anscombe correction for zeros)
    if b == 0 or c == 0:
        # Apply small continuity correction to avoid division by zero
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    else:
        odds_ratio = (a * d) / (b * c)

    data_summary = {
        "table": [[a, b], [c, d]],
        "row1_total": R1,
        "row2_total": R2,
        "col1_total": C1,
        "col2_total": C2,
        "N_total": N,
        "odds_ratio": odds_ratio,
        "p_observed_table": p_observed,
    }

    significance = "significant" if p_value < alpha else "not significant"
    interpretation = (
        f"Fisher's Exact test is {significance} "
        f"(odds ratio = {odds_ratio:.4f}, p = {p_value:.4f}). "
        + (
            "A significant association exists between the two categorical variables."
            if p_value < alpha
            else "No significant association found between the two categorical variables."
        )
    )

    return HypoResult(
        test_name="Fisher's Exact Test",
        statistic=odds_ratio,
        p_value=p_value,
        effect_size=odds_ratio,
        effect_size_name="odds ratio",
        confidence_interval=None,
        degrees_of_freedom=None,
        sample_sizes=N,
        alpha=alpha,
        alternative=alternative,
        interpretation=interpretation,
        data_summary=data_summary,
    )
