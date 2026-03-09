"""
Data utilities and preprocessing transformations.

>>> from hypotestx.utils import coerce_numeric, standardize
"""
from .data_utils import (
    coerce_numeric, detect_missing, drop_missing,
    group_by, split_groups, validate_sample_data,
    summary_table, are_paired,
)
from .preprocessing import (
    standardize, normalize, winsorize,
    log_transform, rank_transform, center,
    robust_scale, apply,
)

__all__ = [
    "coerce_numeric", "detect_missing", "drop_missing",
    "group_by", "split_groups", "validate_sample_data",
    "summary_table", "are_paired",
    "standardize", "normalize", "winsorize",
    "log_transform", "rank_transform", "center",
    "robust_scale", "apply",
]
