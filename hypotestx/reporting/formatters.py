"""
Formatting helpers for statistical reports.

Provides
--------
apa_stat(test_name, statistic, df, p, n)    -> APA-style inline citation string
format_p(p, threshold)                       -> formatted p-value string
format_ci(lower, upper, level)               -> CI string
format_effect(name, value, magnitude)        -> effect size string
effect_interpretation_table()                -> str table of Cohen conventions
"""
from typing import Optional, Union


# ---------------------------------------------------------------------------
# P-value formatting
# ---------------------------------------------------------------------------

def format_p(p: float, threshold: float = 0.001) -> str:
    """
    Format a p-value for display.

    Returns '< 0.001', '0.0123', etc.
    """
    if p < threshold:
        return f"< {threshold}"
    return f"{p:.3f}"


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------

def format_ci(
    lower: float,
    upper: float,
    level: float = 0.95,
    decimal_places: int = 3,
) -> str:
    """
    Format a confidence interval.

    Example: '95% CI [1.234, 5.678]'
    """
    pct = int(level * 100)
    fmt = f".{decimal_places}f"
    return f"{pct}% CI [{lower:{fmt}}, {upper:{fmt}}]"


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def format_effect(name: str, value: float, magnitude: Optional[str] = None) -> str:
    """
    Format an effect size for display.

    Example: "Cohen's d = 0.456 (medium)"
    """
    label = f"{name} = {value:.3f}"
    if magnitude:
        label += f" ({magnitude})"
    return label


# ---------------------------------------------------------------------------
# APA-style test statistic citations
# ---------------------------------------------------------------------------

def apa_stat(
    test_name: str,
    statistic: float,
    df: Optional[Union[int, tuple]] = None,
    p: Optional[float] = None,
    n: Optional[Union[int, tuple]] = None,
    effect_name: Optional[str] = None,
    effect_value: Optional[float] = None,
) -> str:
    """
    Generate an APA-style inline statistics citation string.

    Parameters
    ----------
    test_name    : e.g. 't', 'F', 'chi2', 'chi-square', 'r', 'U'
    statistic    : test statistic value
    df           : degrees of freedom (int or tuple for F)
    p            : p-value
    n            : sample size(s)
    effect_name  : e.g. "d", "r", "eta-squared"
    effect_value : effect size value

    Returns
    -------
    str : e.g. 't(29) = 2.34, p = 0.026, d = 0.43 (small)'

    Examples
    --------
    >>> apa_stat('t', 2.34, df=29, p=0.026, effect_name='d', effect_value=0.43)
    't(29) = 2.34, p = 0.026, d = 0.43'
    """
    # Symbol
    symbols = {
        "t": "t", "f": "F", "chi2": "chi2", "chi-square": "chi2",
        "r": "r", "u": "U", "w": "W", "h": "H", "z": "z",
    }
    sym = symbols.get(test_name.lower(), test_name)

    # df part
    if df is not None:
        if isinstance(df, (list, tuple)):
            df_str = "(" + ", ".join(str(d) for d in df) + ")"
        else:
            df_str = f"({df})"
    else:
        df_str = ""

    parts = [f"{sym}{df_str} = {statistic:.2f}"]

    if n is not None:
        if isinstance(n, (list, tuple)):
            parts.append(f"N = {sum(n)}")
        else:
            parts.append(f"n = {n}")

    if p is not None:
        parts.append(f"p {_p_apa(p)}")

    if effect_name is not None and effect_value is not None:
        parts.append(f"{effect_name} = {effect_value:.3f}")

    return ", ".join(parts)


def _p_apa(p: float) -> str:
    """APA format for p: '< .001' or '= .023' etc."""
    if p < 0.001:
        return "< .001"
    return f"= {p:.3f}".replace("0.", ".")


# ---------------------------------------------------------------------------
# Cohen's convention table
# ---------------------------------------------------------------------------

def effect_interpretation_table() -> str:
    """
    Return a formatted reference table of Cohen's effect-size conventions.
    """
    rows = [
        ("Measure",         "Small",  "Medium", "Large",  "Notes"),
        ("Cohen's d",       "0.20",   "0.50",   "0.80",   "t-tests"),
        ("Cohen's f",       "0.10",   "0.25",   "0.40",   "ANOVA"),
        ("Cohen's w",       "0.10",   "0.30",   "0.50",   "chi-square"),
        ("Pearson r",       "0.10",   "0.30",   "0.50",   "correlation"),
        ("Eta-squared",     "0.01",   "0.06",   "0.14",   "ANOVA"),
        ("Rank-biserial r", "0.10",   "0.30",   "0.50",   "Mann-Whitney"),
        ("Cramer's V",      "0.10",   "0.30",   "0.50",   "chi-square (df>1)"),
        ("Phi",             "0.10",   "0.30",   "0.50",   "chi-square (df=1)"),
    ]

    col_w = [max(len(r[i]) for r in rows) + 2 for i in range(5)]
    sep   = "+" + "+".join("-" * w for w in col_w) + "+"

    lines = [sep]
    for i, row in enumerate(rows):
        line = "|" + "|".join(f" {row[j]:<{col_w[j] - 1}}" for j in range(5)) + "|"
        lines.append(line)
        lines.append(sep)

    return "\n".join(lines)


__all__ = [
    "format_p", "format_ci", "format_effect",
    "apa_stat", "effect_interpretation_table",
]
