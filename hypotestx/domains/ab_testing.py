"""
Domain-specific helpers for A/B testing.

Planned for v0.2.0.  Currently provides a thin wrapper around parametric
and proportion tests that accepts the typical A/B test nomenclature
(control / treatment, conversion rate, etc.).
"""

from __future__ import annotations

from typing import Any, Sequence


def ab_test_proportions(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Any:
    """
    Compare two conversion rates using a z-test for proportions.

    Returns a :class:`~hypotestx.core.result.HypoResult`.
    """
    from ..tests.categorical import chi_square_test  # noqa: F401

    table = [
        [float(control_successes), float(control_n - control_successes)],
        [float(treatment_successes), float(treatment_n - treatment_successes)],
    ]
    return chi_square_test(table, alpha=alpha)


def ab_test_means(
    control: Sequence[float],
    treatment: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Any:
    """
    Compare means of two groups using Welch's t-test.

    Returns a :class:`~hypotestx.core.result.HypoResult`.
    """
    from ..tests.parametric import two_sample_ttest  # noqa: F401

    return two_sample_ttest(
        list(control),
        list(treatment),
        alpha=alpha,
        alternative=alternative,
    )


__all__ = ["ab_test_proportions", "ab_test_means"]
