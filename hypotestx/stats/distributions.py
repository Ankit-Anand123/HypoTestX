"""
Convenience re-exports of math.distributions for the stats sub-package.

This module re-exports the distribution classes so that users can import
them directly from ``hypotestx.stats.distributions``.
"""

from ..math.distributions import ChiSquare, F, Normal, StudentT  # noqa: F401

__all__ = ["Normal", "StudentT", "ChiSquare", "F"]
