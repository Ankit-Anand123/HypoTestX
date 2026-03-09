"""
Descriptive statistics and bootstrap resampling.

>>> from hypotestx.stats import describe, bootstrap_ci
"""
from .descriptive import (
    DescriptiveStats, describe,
    five_number_summary, detect_outliers,
    frequency_table, compare_groups,
)
from .bootstrap import (
    bootstrap_ci, bootstrap_two_sample_ci, bootstrap_mean_ci,
    bootstrap_test, permutation_test,
)

__all__ = [
    "DescriptiveStats", "describe",
    "five_number_summary", "detect_outliers",
    "frequency_table", "compare_groups",
    "bootstrap_ci", "bootstrap_two_sample_ci", "bootstrap_mean_ci",
    "bootstrap_test", "permutation_test",
]
