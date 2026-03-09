"""
Regression-based hypothesis tests.

Planned for v0.3.0.  Currently provides:

  - LinearRegressionTest   — overall F-test and coefficient t-tests
  - LogisticRegressionTest — likelihood-ratio test (planned)

All calculations use the core math layer (no sklearn / statsmodels).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

from ..math.statistics import mean, std, variance, covariance, correlation
from ..math.basic import sqrt
from ..core.result import HypoResult


# ---------------------------------------------------------------------------
# Simple Ordinary-Least-Squares helpers
# ---------------------------------------------------------------------------

def _ols_simple(
    x: List[float],
    y: List[float],
) -> Tuple[float, float, List[float]]:
    """
    Fit y = a + b*x using OLS.

    Returns (intercept, slope, residuals).
    """
    n   = len(x)
    xm  = mean(x)
    ym  = mean(y)
    ss_xy = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    ss_xx = sum((xi - xm) ** 2 for xi in x)
    b = ss_xy / ss_xx if ss_xx != 0 else 0.0
    a = ym - b * xm
    residuals = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    return a, b, residuals


def _ols_f_stat(
    y: List[float],
    y_pred: List[float],
    k: int,
) -> Tuple[float, int, int]:
    """
    F-statistic for overall model fit and associated degrees of freedom.

    Parameters
    ----------
    y      : observed values
    y_pred : fitted values
    k      : number of predictors (not counting intercept)

    Returns (F_statistic, df_model, df_residual)
    """
    n   = len(y)
    ym  = mean(y)
    ss_tot = sum((yi - ym) ** 2 for yi in y)
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred))
    ss_reg = ss_tot - ss_res

    df_model   = k
    df_resid   = n - k - 1

    if df_model <= 0 or df_resid <= 0:
        return float("nan"), df_model, df_resid

    ms_reg = ss_reg / df_model
    ms_res = ss_res / df_resid if df_resid > 0 else float("nan")

    if ms_res == 0 or ms_res != ms_res:
        return float("nan"), df_model, df_resid

    return ms_reg / ms_res, df_model, df_resid


# ---------------------------------------------------------------------------
# Linear Regression Test
# ---------------------------------------------------------------------------

class LinearRegressionTest:
    """
    Test for a significant linear relationship between a predictor x
    and a numeric response y.

    Parameters
    ----------
    x     : predictor variable (numeric list)
    y     : response variable (numeric list)
    alpha : significance level (default 0.05)

    Examples
    --------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2.1, 4.0, 5.9, 8.2, 9.8]
    >>> test = LinearRegressionTest(x, y)
    >>> result = test.fit()
    >>> print(result.test_name)
    Linear Regression F-test
    """

    def __init__(
        self,
        x: Sequence[float],
        y: Sequence[float],
        alpha: float = 0.05,
    ) -> None:
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.alpha = alpha

    def fit(self) -> HypoResult:
        """
        Fit the model and return a :class:`HypoResult` for the overall F-test.

        Effect size is R-squared.
        """
        from ..math.distributions import F as FDist

        x, y = self.x, self.y
        n   = len(x)

        if n < 3:
            raise ValueError("LinearRegressionTest requires at least 3 observations")
        if len(y) != n:
            raise ValueError("x and y must have the same length")

        intercept, slope, residuals = _ols_simple(x, y)
        y_pred = [intercept + slope * xi for xi in x]

        f_stat, df1, df2 = _ols_f_stat(y, y_pred, k=1)

        if df1 <= 0 or df2 <= 0 or f_stat != f_stat:
            p_value = float("nan")
            r_sq    = float("nan")
        else:
            f_dist  = FDist(df1, df2)
            p_value = 1.0 - f_dist.cdf(f_stat)

            ym    = mean(y)
            ss_tot = sum((yi - ym) ** 2 for yi in y)
            ss_res = sum(r ** 2 for r in residuals)
            r_sq   = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

        return HypoResult(
            test_name="Linear Regression F-test",
            statistic=f_stat,
            p_value=p_value,
            alpha=self.alpha,
            effect_size=r_sq,
            additional={
                "intercept":  intercept,
                "slope":      slope,
                "df_model":   df1,
                "df_residual": df2,
                "r_squared":  r_sq,
            },
        )

    @property
    def r_squared(self) -> float:
        """Pearson R^2 as a quick shortcut (does not store the full model)."""
        r = correlation(self.x, self.y)
        return r ** 2

    @property
    def coefficients(self) -> Dict[str, float]:
        """Return {'intercept': a, 'slope': b}."""
        a, b, _ = _ols_simple(self.x, self.y)
        return {"intercept": a, "slope": b}


# ---------------------------------------------------------------------------
# Logistic Regression Test  (planned)
# ---------------------------------------------------------------------------

class LogisticRegressionTest:
    """
    Planned for v0.4.0: likelihood-ratio test for logistic regression.

    Currently raises :exc:`NotImplementedError`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "LogisticRegressionTest is planned for v0.4.0. "
            "Track progress at https://github.com/hypotestx/hypotestx/issues."
        )


__all__ = [
    "LinearRegressionTest",
    "LogisticRegressionTest",
]
