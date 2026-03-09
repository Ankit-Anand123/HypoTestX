"""
Core infrastructure: result objects, exceptions, validators, assumptions.
"""
from .result import HypoResult
from .exceptions import (
    HypoTestXError, InsufficientDataError, AssumptionViolationError,
    InvalidAlternativeError, ParseError, UnsupportedTestError, DataFormatError,
)
from .assumptions import (
    shapiro_wilk, levene_test, bartlett_test, jarque_bera,
    check_normality, check_equal_variances,
)

__all__ = [
    "HypoResult",
    "HypoTestXError", "InsufficientDataError", "AssumptionViolationError",
    "InvalidAlternativeError", "ParseError", "UnsupportedTestError", "DataFormatError",
    "shapiro_wilk", "levene_test", "bartlett_test", "jarque_bera",
    "check_normality", "check_equal_variances",
]
