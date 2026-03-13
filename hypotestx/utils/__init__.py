"""
Data utilities and preprocessing transformations.

>>> from hypotestx.utils import coerce_numeric, standardize
"""

from .data_utils import (
    are_paired,
    coerce_numeric,
    detect_missing,
    drop_missing,
    group_by,
    split_groups,
    summary_table,
    validate_sample_data,
)
from .preprocessing import (
    apply,
    center,
    log_transform,
    normalize,
    rank_transform,
    robust_scale,
    standardize,
    winsorize,
)

__all__ = [
    "coerce_numeric",
    "detect_missing",
    "drop_missing",
    "group_by",
    "split_groups",
    "validate_sample_data",
    "summary_table",
    "are_paired",
    "standardize",
    "normalize",
    "winsorize",
    "log_transform",
    "rank_transform",
    "center",
    "robust_scale",
    "apply",
]
