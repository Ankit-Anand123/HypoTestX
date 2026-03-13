"""
Mathematical functions and distributions for HypoTestX

This module provides pure Python implementations of mathematical functions,
statistical operations, and probability distributions.
"""

# Basic mathematical operations
from .basic import (
    PI,
    E,
    abs_value,
    combination,
    exp,
    factorial,
    ln,
    log,
    power,
    sign,
    sqrt,
)

# Probability distributions
from .distributions import ChiSquare, Distribution, F, Normal, StudentT

# Linear algebra operations
from .linear_algebra import (
    Matrix,
    eigenvalues,
    eigenvectors,
    matrix_inverse,
    matrix_multiply,
    matrix_transpose,
    qr_decomposition,
    svd_decomposition,
    vector_dot,
    vector_norm,
)

# Special mathematical functions
from .special import beta, beta_incomplete, erf, gamma, gamma_incomplete

# Statistical functions
from .statistics import (
    correlation,
    covariance,
    iqr,
    kurtosis,
    mad,
    mean,
    median,
    mode,
    percentile,
    quartiles,
    range_stat,
    skewness,
    std,
    trimmed_mean,
    variance,
)

__all__ = [
    # Basic math
    "abs_value",
    "sqrt",
    "exp",
    "ln",
    "log",
    "power",
    "factorial",
    "combination",
    "sign",
    "PI",
    "E",
    # Statistics
    "mean",
    "median",
    "mode",
    "variance",
    "std",
    "covariance",
    "correlation",
    "skewness",
    "kurtosis",
    "percentile",
    "quartiles",
    "iqr",
    "range_stat",
    "mad",
    "trimmed_mean",
    # Special functions
    "gamma",
    "beta",
    "erf",
    "gamma_incomplete",
    "beta_incomplete",
    # Distributions
    "Distribution",
    "Normal",
    "StudentT",
    "ChiSquare",
    "F",
    # Linear algebra
    "Matrix",
    "vector_dot",
    "vector_norm",
    "matrix_multiply",
    "matrix_transpose",
    "matrix_inverse",
    "eigenvalues",
    "eigenvectors",
    "qr_decomposition",
    "svd_decomposition",
]
