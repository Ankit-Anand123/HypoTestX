"""
TestSuite — run multiple HypoTestX tests and aggregate results.

Planned feature; file reserved for v0.2.0.
"""
from __future__ import annotations
from typing import Callable, List, Optional

from .result import HypoResult


class TestSuite:
    """
    Collect and run multiple hypothesis tests, returning aggregated results.

    Planned for v0.2.0.
    """

    # Prevent pytest from treating this class as a test collector
    __test__ = False

    def __init__(self, name: str = "TestSuite") -> None:
        self.name = name
        self._tests: List[Callable[[], HypoResult]] = []
        self._results: List[HypoResult] = []

    def add(self, test_fn: Callable[[], HypoResult]) -> "TestSuite":
        """Register a zero-argument callable that returns a HypoResult."""
        self._tests.append(test_fn)
        return self

    def run(self) -> List[HypoResult]:
        """Execute all registered tests and return the results."""
        self._results = [fn() for fn in self._tests]
        return self._results

    @property
    def n_significant(self) -> int:
        """Number of significant results (after running)."""
        return sum(1 for r in self._results if r.is_significant)

    def summary(self) -> str:
        total = len(self._results)
        sig   = self.n_significant
        lines = [f"TestSuite '{self.name}': {total} tests, {sig} significant"]
        for i, r in enumerate(self._results, 1):
            status = "SIGNIFICANT" if r.is_significant else "ns"
            lines.append(f"  [{i}] {r.test_name}: p={r.p_value:.4f} ({status})")
        return "\n".join(lines)


__all__ = ["TestSuite"]
