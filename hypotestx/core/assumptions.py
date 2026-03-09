"""
Statistical assumption-checking functions.

Provides
--------
shapiro_wilk(data, alpha)          : Shapiro-Wilk normality test (Royston 1992 approximation)
levene_test(*groups, alpha)        : Levene (Brown-Forsythe) equality-of-variances test
bartlett_test(*groups, alpha)      : Bartlett equality-of-variances test (assumes normality)
jarque_bera(data, alpha)           : Jarque-Bera normality test (skewness + kurtosis)
check_normality(data, alpha)       : -> (is_normal: bool, HypoResult)
check_equal_variances(*groups)     : -> (are_equal: bool, HypoResult)
"""
from typing import List, Tuple, Dict
from ..math.statistics import mean, variance, std, median, skewness, kurtosis
from ..math.basic import sqrt, ln, exp, abs_value
from ..math.distributions import Normal, ChiSquare
from ..core.result import HypoResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _poly(coeffs: list, x: float) -> float:
    """Evaluate polynomial: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ..."""
    return sum(c * x ** i for i, c in enumerate(coeffs))


def _blom_expected(n: int) -> List[float]:
    """
    Expected standard normal order statistics via Blom (1958) approximation:
      m_i = Phi^{-1}((i - 3/8) / (n + 1/4))
    """
    phi = Normal(0, 1)
    return [phi.ppf((i - 3 / 8) / (n + 1 / 4)) for i in range(1, n + 1)]


def _sw_pvalue(W: float, n: int) -> float:
    """
    Approximate p-value for the Shapiro-Wilk W statistic.
    Uses the Royston (1992) normal-transformation approximation.
    A SMALL p-value means evidence AGAINST normality.
    """
    if W >= 1.0 - 1e-9:
        return 1.0
    if W <= 0.0:
        return 0.0

    y = ln(1.0 - W)

    if n <= 11:
        # Polynomial in n
        mu    = _poly([0.0, -0.6714, 0.025054, -6.714e-4], n)
        sigma = exp(_poly([2.1349, -0.63695, 0.062767, -0.0020322], n))
    else:
        # Polynomial in ln(n) / ln(ln(n))
        ln_n   = ln(float(n))
        mu     = _poly([-1.5861, -0.31082, -0.083751, 0.0038915], ln_n)
        ln_ln_n = ln(ln_n) if ln_n > 1.0 else 0.0
        sigma  = exp(_poly([-1.5614, -0.5836, 0.1477], ln_ln_n))

    if sigma <= 0:
        return 0.5
    z = (y - mu) / sigma
    p = 1.0 - Normal(0, 1).cdf(z)
    return min(max(p, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Shapiro-Wilk
# ---------------------------------------------------------------------------

def shapiro_wilk(data: List[float], alpha: float = 0.05) -> HypoResult:
    """
    Shapiro-Wilk normality test.

    Uses normalized Blom (1958) expected order statistics as W coefficients
    and the Royston (1992) approximation for the p-value.

    Parameters
    ----------
    data  : numeric list, n >= 3
    alpha : significance level (default 0.05)

    Returns
    -------
    HypoResult
        statistic = W (0..1), higher = more normal
        is_significant = True  means evidence AGAINST normality (reject H0)
    """
    data_sorted = sorted(float(x) for x in data)
    n = len(data_sorted)
    if n < 3:
        raise ValueError("Shapiro-Wilk requires at least 3 observations")

    mu_d = mean(data_sorted)
    S2   = sum((x - mu_d) ** 2 for x in data_sorted)

    if S2 == 0.0:
        # Constant data — not normal (or trivially normal for a degenerate sense)
        W       = 1.0
        p_value = 0.0
    else:
        m     = _blom_expected(n)
        ms    = sqrt(sum(mi ** 2 for mi in m))
        a     = [mi / ms for mi in m]        # a_i = normalized expected order statistics

        k = n // 2
        # Pairs sum: a[n-1-i] * (x_(n-i) - x_(i+1))  [1-based: a_n*(x_n-x_1) + a_{n-1}*(x_{n-1}-x_2) + ...]
        numerator = sum(a[n - 1 - i] * (data_sorted[n - 1 - i] - data_sorted[i])
                        for i in range(k))
        W       = min(max((numerator ** 2) / S2, 0.0), 1.0)
        p_value = _sw_pvalue(W, n)

    is_normal = p_value >= alpha
    if is_normal:
        interp = ("No significant departure from normality detected "
                  f"(W = {W:.4f}, p = {p_value:.4f} >= {alpha}).")
    else:
        interp = ("Significant departure from normality detected "
                  f"(W = {W:.4f}, p = {p_value:.4f} < {alpha}).")

    return HypoResult(
        test_name="Shapiro-Wilk Normality Test",
        statistic=round(W, 6),
        p_value=p_value,
        alpha=alpha,
        sample_sizes=n,
        assumptions_met={"normality": is_normal},
        interpretation=interp,
        data_summary={"n": n, "W": round(W, 6)},
    )


# ---------------------------------------------------------------------------
# Levene (Brown-Forsythe, median-based)
# ---------------------------------------------------------------------------

def levene_test(*groups: List[float], alpha: float = 0.05) -> HypoResult:
    """
    Levene's test for equality of variances (Brown-Forsythe variant).

    Uses median-based absolute deviations, which is robust to non-normality.

    Parameters
    ----------
    *groups : two or more numeric lists
    alpha   : significance level (default 0.05)

    Returns
    -------
    HypoResult
        statistic = F (Brown-Forsythe)
        is_significant = True  means UNEQUAL variances detected
    """
    from ..math.distributions import F as FDist

    groups_clean = [list(map(float, g)) for g in groups]
    k = len(groups_clean)
    if k < 2:
        raise ValueError("Levene test requires at least 2 groups")
    for i, g in enumerate(groups_clean):
        if len(g) < 2:
            raise ValueError(f"Group {i + 1} has fewer than 2 observations")

    ns  = [len(g) for g in groups_clean]
    N   = sum(ns)

    # Absolute deviations from group medians
    zij     = [[abs_value(x - median(g)) for x in g] for g in groups_clean]
    z_bar_j = [mean(z) for z in zij]
    z_bar   = sum(ns[j] * z_bar_j[j] for j in range(k)) / N

    num = sum(ns[j] * (z_bar_j[j] - z_bar) ** 2 for j in range(k)) / (k - 1)
    den = sum(
        sum((z - z_bar_j[j]) ** 2 for z in zij[j]) for j in range(k)
    ) / (N - k)

    if den == 0.0:
        F_stat  = 0.0
        p_value = 1.0
    else:
        F_stat  = num / den
        p_value = 1.0 - FDist(k - 1, N - k).cdf(F_stat)

    equal_var = p_value >= alpha
    if equal_var:
        interp = (f"Equal variances assumed (F = {F_stat:.4f}, p = {p_value:.4f} >= {alpha}).")
    else:
        interp = (f"Unequal variances detected (F = {F_stat:.4f}, p = {p_value:.4f} < {alpha}).")

    return HypoResult(
        test_name="Levene Test for Equal Variances",
        statistic=round(F_stat, 6),
        p_value=p_value,
        alpha=alpha,
        degrees_of_freedom=(k - 1, N - k),
        sample_sizes=tuple(ns),
        assumptions_met={"equal_variances": equal_var},
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Bartlett
# ---------------------------------------------------------------------------

def bartlett_test(*groups: List[float], alpha: float = 0.05) -> HypoResult:
    """
    Bartlett's test for equality of variances.

    More powerful than Levene when normality holds, but sensitive to
    departures from normality.  Prefer Levene for non-normal data.

    Parameters
    ----------
    *groups : two or more numeric lists
    alpha   : significance level

    Returns
    -------
    HypoResult
        statistic = B (chi-square)
        is_significant = True  means UNEQUAL variances detected
    """
    groups_clean = [list(map(float, g)) for g in groups]
    k = len(groups_clean)
    if k < 2:
        raise ValueError("Bartlett test requires at least 2 groups")

    ns    = [len(g) for g in groups_clean]
    N     = sum(ns)
    vars_ = [variance(g, ddof=1) for g in groups_clean]

    if any(v <= 0 for v in vars_):
        raise ValueError("One or more groups have zero variance")

    sp2 = sum((ns[j] - 1) * vars_[j] for j in range(k)) / (N - k)

    numer  = (N - k) * ln(sp2) - sum((ns[j] - 1) * ln(vars_[j]) for j in range(k))
    denom  = 1.0 + (1.0 / (3.0 * (k - 1))) * (
        sum(1.0 / (ns[j] - 1) for j in range(k)) - 1.0 / (N - k)
    )
    B       = numer / denom
    p_value = 1.0 - ChiSquare(k - 1).cdf(B)

    equal_var = p_value >= alpha
    if equal_var:
        interp = (f"Equal variances assumed (B = {B:.4f}, p = {p_value:.4f} >= {alpha}).")
    else:
        interp = (f"Unequal variances detected (B = {B:.4f}, p = {p_value:.4f} < {alpha}).")

    return HypoResult(
        test_name="Bartlett Test for Equal Variances",
        statistic=round(B, 6),
        p_value=p_value,
        alpha=alpha,
        degrees_of_freedom=k - 1,
        sample_sizes=tuple(ns),
        assumptions_met={"equal_variances": equal_var},
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Jarque-Bera
# ---------------------------------------------------------------------------

def jarque_bera(data: List[float], alpha: float = 0.05) -> HypoResult:
    """
    Jarque-Bera test for normality based on skewness and excess kurtosis.

    JB = n/6 * (S^2 + K^2/4)  ~  chi-squared(2) under H0

    where S = sample skewness, K = sample excess kurtosis.
    Fast and easy to compute; less powerful than Shapiro-Wilk for small n.

    Parameters
    ----------
    data  : numeric list, n >= 4
    alpha : significance level

    Returns
    -------
    HypoResult
        statistic = JB
        is_significant = True  means evidence AGAINST normality
    """
    data = [float(x) for x in data]
    n    = len(data)
    if n < 4:
        raise ValueError("Jarque-Bera requires at least 4 observations")

    S  = skewness(data)
    K  = kurtosis(data)   # excess kurtosis (already -3 corrected in statistics.py)
    JB = (n / 6.0) * (S ** 2 + (K ** 2) / 4.0)

    p_value   = 1.0 - ChiSquare(2).cdf(JB)
    is_normal = p_value >= alpha

    if is_normal:
        interp = (f"No significant departure from normality "
                  f"(JB = {JB:.4f}, p = {p_value:.4f} >= {alpha}).")
    else:
        interp = (f"Significant departure from normality "
                  f"(JB = {JB:.4f}, skewness = {S:.4f}, excess kurtosis = {K:.4f}, "
                  f"p = {p_value:.4f} < {alpha}).")

    return HypoResult(
        test_name="Jarque-Bera Normality Test",
        statistic=round(JB, 6),
        p_value=p_value,
        alpha=alpha,
        sample_sizes=n,
        assumptions_met={"normality": is_normal},
        interpretation=interp,
        data_summary={"n": n, "skewness": round(S, 4), "excess_kurtosis": round(K, 4)},
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def check_normality(
    data: List[float],
    alpha: float = 0.05,
    method: str = "shapiro-wilk",
) -> Tuple[bool, HypoResult]:
    """
    Return (is_normal, HypoResult).

    Parameters
    ----------
    data   : numeric list
    alpha  : significance level
    method : 'shapiro-wilk' (default) or 'jarque-bera'
    """
    method = method.lower().replace("_", "-")
    if method in ("shapiro-wilk", "sw"):
        result = shapiro_wilk(data, alpha=alpha)
    elif method in ("jarque-bera", "jb"):
        result = jarque_bera(data, alpha=alpha)
    else:
        raise ValueError("method must be 'shapiro-wilk' or 'jarque-bera'")
    return result.p_value >= alpha, result


def check_equal_variances(
    *groups: List[float],
    alpha: float = 0.05,
    method: str = "levene",
) -> Tuple[bool, HypoResult]:
    """
    Return (are_equal, HypoResult).

    Parameters
    ----------
    *groups : two or more numeric lists
    alpha   : significance level
    method  : 'levene' (default, robust) or 'bartlett' (assumes normality)
    """
    method = method.lower()
    if method == "levene":
        result = levene_test(*groups, alpha=alpha)
    elif method == "bartlett":
        result = bartlett_test(*groups, alpha=alpha)
    else:
        raise ValueError("method must be 'levene' or 'bartlett'")
    return result.p_value >= alpha, result


__all__ = [
    "shapiro_wilk", "levene_test", "bartlett_test", "jarque_bera",
    "check_normality", "check_equal_variances",
]
