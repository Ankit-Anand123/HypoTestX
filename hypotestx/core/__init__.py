"""
Core infrastructure: result objects, exceptions, validators, assumptions.
"""

from .assumptions import (
    bartlett_test,
    check_equal_variances,
    check_normality,
    jarque_bera,
    levene_test,
    shapiro_wilk,
)
from .exceptions import (
    AssumptionViolationError,
    DataFormatError,
    HypoTestXError,
    InsufficientDataError,
    InvalidAlternativeError,
    ParseError,
    UnsupportedTestError,
)
from .result import HypoResult

__all__ = [
    "HypoResult",
    "HypoTestXError",
    "InsufficientDataError",
    "AssumptionViolationError",
    "InvalidAlternativeError",
    "ParseError",
    "UnsupportedTestError",
    "DataFormatError",
    "shapiro_wilk",
    "levene_test",
    "bartlett_test",
    "jarque_bera",
    "check_normality",
    "check_equal_variances",
]
