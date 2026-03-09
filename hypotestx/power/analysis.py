"""
Post-hoc statistical power analysis.

Provides
--------
power_ttest_one_sample(effect_size, n, alpha, alternative)
power_ttest_two_sample(effect_size, n1, n2, alpha, alternative)
power_anova(effect_size, n_per_group, k, alpha)
power_chi_square(effect_size, n, df, alpha)
power_correlation(r, n, alpha, alternative)
"""
from typing import Optional
from ..math.basic import sqrt, abs_value, ln
from ..math.distributions import Normal


# ---------------------------------------------------------------------------
# Internal: tail probability helpers
# ---------------------------------------------------------------------------

def _power_from_ncp(ncp: float, alpha: float, alternative: str) -> float:
    """
    Approximate power using normal distribution with non-centrality parameter.
    Uses z-test approximation which is accurate for moderate-to-large n.
    """
    z = Normal(0, 1)
    if alternative == "two-sided":
        z_crit = z.ppf(1.0 - alpha / 2.0)
        power  = 1.0 - z.cdf(z_crit - ncp) + z.cdf(-z_crit - ncp)
    elif alternative in ("greater", "less"):
        z_crit = z.ppf(1.0 - alpha)
        power  = 1.0 - z.cdf(z_crit - abs_value(ncp))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return min(max(power, 0.0), 1.0)


# ---------------------------------------------------------------------------
# One-sample t-test power
# ---------------------------------------------------------------------------

def power_ttest_one_sample(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Post-hoc power for a one-sample t-test.

    Parameters
    ----------
    effect_size : Cohen's d  (|mean - mu0| / sigma)
    n           : sample size
    alpha       : significance level (default 0.05)
    alternative : 'two-sided', 'greater', or 'less'

    Returns
    -------
    float : estimated statistical power (0..1)

    Notes
    -----
    Uses normal approximation of the non-central t-distribution.
    NCP = d * sqrt(n).
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    ncp = abs_value(effect_size) * sqrt(n)
    return _power_from_ncp(ncp, alpha, alternative)


# ---------------------------------------------------------------------------
# Two-sample t-test power
# ---------------------------------------------------------------------------

def power_ttest_two_sample(
    effect_size: float,
    n1: int,
    n2: Optional[int] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Post-hoc power for an independent-samples t-test.

    Parameters
    ----------
    effect_size : Cohen's d  ((mean1 - mean2) / pooled_sigma)
    n1          : first group size
    n2          : second group size (default = n1, i.e. balanced design)
    alpha       : significance level
    alternative : 'two-sided', 'greater', or 'less'

    Returns
    -------
    float : estimated statistical power (0..1)

    Notes
    -----
    NCP = d * sqrt(n1*n2/(n1+n2)).
    """
    if n2 is None:
        n2 = n1
    if n1 < 2 or n2 < 2:
        raise ValueError("Both group sizes must be at least 2")
    n_harm = (n1 * n2) / (n1 + n2)   # harmonic-mean-like term
    ncp    = abs_value(effect_size) * sqrt(n_harm)
    return _power_from_ncp(ncp, alpha, alternative)


# ---------------------------------------------------------------------------
# Paired t-test power
# ---------------------------------------------------------------------------

def power_ttest_paired(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Post-hoc power for a paired t-test.

    Equivalent to one-sample power on the differences.
    NCP = d * sqrt(n).
    """
    return power_ttest_one_sample(effect_size, n, alpha=alpha, alternative=alternative)


# ---------------------------------------------------------------------------
# One-way ANOVA power
# ---------------------------------------------------------------------------

def power_anova(
    effect_size: float,
    n_per_group: int,
    k: int,
    alpha: float = 0.05,
) -> float:
    """
    Post-hoc power for a one-way ANOVA using Cohen's f.

    Parameters
    ----------
    effect_size : Cohen's f  (sigma_means / sigma_within)
                  Conventions: small=0.10, medium=0.25, large=0.40
    n_per_group : observations per group (balanced design)
    k           : number of groups
    alpha       : significance level

    Returns
    -------
    float : estimated statistical power (0..1)

    Notes
    -----
    NCP = f * sqrt(k * n_per_group).
    Uses chi-square approximation.
    """
    if k < 2:
        raise ValueError("ANOVA requires at least 2 groups")
    if n_per_group < 2:
        raise ValueError("n_per_group must be at least 2")

    N   = k * n_per_group
    ncp = (effect_size ** 2) * N          # lambda = f^2 * N

    from ..math.distributions import ChiSquare
    # Approximate: compare non-central chi^2 tail to central chi^2 critical value
    df_between = k - 1
    chi2_crit  = ChiSquare(df_between).ppf(1.0 - alpha)

    # Power = P(chi^2(df, ncp) > chi2_crit)
    # Use shifted normal approximation: chi^2(df,ncp) ~ N(df+ncp, 2*(df+2*ncp))
    from ..math.basic import sqrt as _sqrt
    mean_nc = df_between + ncp
    std_nc  = _sqrt(2 * (df_between + 2 * ncp))
    if std_nc == 0:
        return 0.0
    z_power = (chi2_crit - mean_nc) / std_nc
    power   = 1.0 - Normal(0, 1).cdf(z_power)
    return min(max(power, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Chi-square test power
# ---------------------------------------------------------------------------

def power_chi_square(
    effect_size: float,
    n: int,
    df: int,
    alpha: float = 0.05,
) -> float:
    """
    Post-hoc power for a chi-square test using Cohen's w.

    Parameters
    ----------
    effect_size : Cohen's w  (= Cramer's V * sqrt(min(rows,cols)-1) for cont. tables)
                  Conventions: small=0.10, medium=0.30, large=0.50
    n           : total sample size
    df          : degrees of freedom of the chi-square statistic
    alpha       : significance level

    Returns
    -------
    float : estimated power (0..1)
    """
    from ..math.distributions import ChiSquare
    ncp       = (effect_size ** 2) * n
    chi2_crit = ChiSquare(df).ppf(1.0 - alpha)

    mean_nc = df + ncp
    std_nc  = sqrt(2.0 * (df + 2.0 * ncp))
    if std_nc == 0:
        return 0.0
    z_power = (chi2_crit - mean_nc) / std_nc
    power   = 1.0 - Normal(0, 1).cdf(z_power)
    return min(max(power, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Correlation power
# ---------------------------------------------------------------------------

def power_correlation(
    r: float,
    n: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Post-hoc power for a Pearson/Spearman correlation test.

    Uses Fisher's z-transform approximation.
    NCP = |z_r| * sqrt(n - 3)  where z_r = 0.5 * ln((1+r)/(1-r)).

    Parameters
    ----------
    r           : effect size (Pearson r, -1..1, 0 excluded)
    n           : sample size
    alpha       : significance level
    alternative : 'two-sided', 'greater', or 'less'
    """
    if abs_value(r) >= 1.0:
        return 1.0
    if n < 4:
        raise ValueError("n must be at least 4 for correlation power")

    z_r = 0.5 * ln((1.0 + r) / (1.0 - r))
    ncp = abs_value(z_r) * sqrt(float(n - 3))
    return _power_from_ncp(ncp, alpha, alternative)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def power_summary(
    test_type: str,
    effect_size: float,
    alpha: float = 0.05,
    n: Optional[int] = None,
    **kwargs,
) -> str:
    """
    Human-readable power summary.

    Parameters
    ----------
    test_type   : 'one_sample_t', 'two_sample_t', 'anova', 'chi_square', 'correlation'
    effect_size : appropriate effect-size measure
    alpha       : significance level
    n           : sample size (first/main group)
    **kwargs    : additional arguments forwarded to the power function
    """
    dispatch = {
        "one_sample_t":  power_ttest_one_sample,
        "paired_t":      power_ttest_paired,
        "two_sample_t":  power_ttest_two_sample,
        "anova":         power_anova,
        "chi_square":    power_chi_square,
        "correlation":   power_correlation,
    }
    if test_type not in dispatch:
        raise ValueError(f"test_type must be one of: {list(dispatch.keys())}")
    if n is None:
        raise ValueError("n is required for power_summary")

    func  = dispatch[test_type]
    power = func(effect_size, n, alpha=alpha, **kwargs)

    lines = [
        f"Power Analysis: {test_type.replace('_', ' ').title()}",
        f"  Effect size : {effect_size:.4f}",
        f"  n           : {n}",
        f"  alpha       : {alpha}",
        f"  Power       : {power:.4f}",
    ]
    if power < 0.80:
        lines.append("  [!] Power < 0.80. Consider increasing sample size.")
    else:
        lines.append("  [ok] Adequate power (>= 0.80).")
    return "\n".join(lines)


__all__ = [
    "power_ttest_one_sample",
    "power_ttest_two_sample",
    "power_ttest_paired",
    "power_anova",
    "power_chi_square",
    "power_correlation",
    "power_summary",
]
