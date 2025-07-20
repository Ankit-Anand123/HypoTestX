"""
Mathematical functions and distributions for HypoTestX

This module provides pure Python implementations of mathematical functions,
statistical operations, and probability distributions.
"""

# Basic mathematical operations
from .basic import (
    abs_value,
    sqrt, 
    exp,
    ln,
    log,
    power,
    factorial,
    combination,
    sign,
    PI,
    E
)

# Statistical functions
from .statistics import (
    mean,
    median,
    mode,
    variance,
    std,
    covariance,
    correlation,
    skewness,
    kurtosis,
    percentile,
    quartiles,
    iqr,
    range_stat,
    mad,
    trimmed_mean
)

# Special mathematical functions
from .special import (
    gamma,
    beta,
    erf,
    gamma_incomplete,
    beta_incomplete
)

# Probability distributions
from .distributions import (
    Distribution,
    Normal,
    StudentT,
    ChiSquare,
    F
)

# Linear algebra operations
from .linear_algebra import (
    Matrix,
    vector_dot,
    vector_norm,
    matrix_multiply,
    matrix_transpose,
    matrix_inverse,
    eigenvalues,
    eigenvectors,
    qr_decomposition,
    svd_decomposition
)

__all__ = [
    # Basic math
    'abs_value', 'sqrt', 'exp', 'ln', 'log', 'power', 'factorial', 
    'combination', 'sign', 'PI', 'E',
    
    # Statistics
    'mean', 'median', 'mode', 'variance', 'std', 'covariance', 
    'correlation', 'skewness', 'kurtosis', 'percentile', 'quartiles',
    'iqr', 'range_stat', 'mad', 'trimmed_mean',
    
    # Special functions
    'gamma', 'beta', 'erf', 'gamma_incomplete', 'beta_incomplete',
    
    # Distributions
    'Distribution', 'Normal', 'StudentT', 'ChiSquare', 'F',
    
    # Linear algebra
    'Matrix', 'vector_dot', 'vector_norm', 'matrix_multiply', 
    'matrix_transpose', 'matrix_inverse', 'eigenvalues', 'eigenvectors',
    'qr_decomposition', 'svd_decomposition'
]