"""
Custom exceptions for HypoTestX
"""


class HypoTestXError(Exception):
    """Base exception for all HypoTestX errors"""
    pass


class InsufficientDataError(HypoTestXError):
    """Raised when the data is too small for a valid test"""

    def __init__(self, message: str = "Not enough data points to perform the test"):
        super().__init__(message)
        self.message = message


class AssumptionViolationError(HypoTestXError):
    """Raised when a required statistical assumption is violated"""

    def __init__(self, assumption: str, message: str = ""):
        full_msg = f"Assumption violated: '{assumption}'"
        if message:
            full_msg += f". {message}"
        super().__init__(full_msg)
        self.assumption = assumption
        self.message = full_msg


class InvalidAlternativeError(HypoTestXError):
    """Raised when an unsupported alternative hypothesis is specified"""

    def __init__(self, alternative: str, valid=None):
        if valid is None:
            valid = ["two-sided", "greater", "less"]
        msg = f"Invalid alternative '{alternative}'. Must be one of {valid}."
        super().__init__(msg)
        self.alternative = alternative


class ParseError(HypoTestXError):
    """Raised when a natural language hypothesis cannot be parsed"""

    def __init__(self, hypothesis: str, reason: str = ""):
        msg = f"Could not parse hypothesis: '{hypothesis}'"
        if reason:
            msg += f". Reason: {reason}"
        super().__init__(msg)
        self.hypothesis = hypothesis


class UnsupportedTestError(HypoTestXError):
    """Raised when the requested test type is not implemented"""

    def __init__(self, test_type: str):
        msg = (
            f"Test type '{test_type}' is not supported or not yet implemented. "
            "Use explicit test functions (e.g., htx.ttest_2samp) instead."
        )
        super().__init__(msg)
        self.test_type = test_type


class DataFormatError(HypoTestXError):
    """Raised when data is in an unexpected or incompatible format"""

    def __init__(self, message: str = "Data is in an unsupported format"):
        super().__init__(message)
        self.message = message
