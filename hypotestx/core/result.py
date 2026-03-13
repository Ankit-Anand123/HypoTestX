"""
HypoResult class for standardized test results
"""

from typing import Any, Dict, List, Optional, Union

from ..utils.formatting import format_effect_size, format_p_value


class HypoResult:
    """Standardized result object for hypothesis tests"""

    def __init__(
        self,
        test_name: str,
        statistic: float,
        p_value: float,
        effect_size: Optional[float] = None,
        effect_size_name: Optional[str] = None,
        confidence_interval: Optional[tuple] = None,
        degrees_of_freedom: Optional[Union[int, tuple]] = None,
        sample_sizes: Optional[Union[int, tuple]] = None,
        assumptions_met: Optional[Dict[str, bool]] = None,
        interpretation: Optional[str] = None,
        data_summary: Optional[Dict[str, Any]] = None,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        **kwargs,
    ):
        self.test_name = test_name
        self.statistic = statistic
        self.p_value = p_value
        self.effect_size = effect_size
        self.effect_size_name = effect_size_name
        self.confidence_interval = confidence_interval
        self.degrees_of_freedom = degrees_of_freedom
        self.sample_sizes = sample_sizes
        self.assumptions_met = assumptions_met or {}
        self.interpretation = interpretation
        self.data_summary = data_summary or {}
        self.alpha = alpha
        self.alternative = alternative
        self.extra_info = kwargs
        # Routing metadata (populated by analyze())
        self.routing_confidence: float = 1.0
        self.routing_source: str = "llm"

    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant"""
        return self.p_value < self.alpha

    @property
    def effect_magnitude(self) -> str:
        """Interpret effect size magnitude using Cohen's conventions for the appropriate measure."""
        if self.effect_size is None:
            return "not calculated"

        es = abs(self.effect_size)
        name = (self.effect_size_name or "").lower()

        # Cohen's d (t-tests)
        if "cohen" in name or name == "d":
            if es < 0.2:
                return "negligible"
            if es < 0.5:
                return "small"
            if es < 0.8:
                return "medium"
            return "large"

        # Pearson / Spearman r, rank-biserial r, point-biserial r
        if name in ("r", "rank-biserial r", "point-biserial r", "spearman rho"):
            if es < 0.10:
                return "negligible"
            if es < 0.30:
                return "small"
            if es < 0.50:
                return "medium"
            return "large"

        # Cramer's V / phi (chi-square family)
        if name in ("cramer's v", "phi", "cohens w", "cohen's w", "w"):
            if es < 0.10:
                return "negligible"
            if es < 0.30:
                return "small"
            if es < 0.50:
                return "medium"
            return "large"

        # Eta-squared / partial eta-squared (ANOVA / Kruskal-Wallis)
        if "eta" in name:
            if es < 0.01:
                return "negligible"
            if es < 0.06:
                return "small"
            if es < 0.14:
                return "medium"
            return "large"

        # Omega-squared
        if "omega" in name:
            if es < 0.01:
                return "negligible"
            if es < 0.06:
                return "small"
            if es < 0.14:
                return "medium"
            return "large"

        # Odds ratio — not on a linear scale; give qualitative label
        if "odds" in name:
            if es < 1.5:
                return "negligible"
            if es < 2.5:
                return "small"
            if es < 4.0:
                return "medium"
            return "large"

        return "unknown scale"

    def summary(self, verbose: bool = False) -> str:
        """Human-readable summary of results"""
        lines = []

        # Header
        header = f"[ {self.test_name} ]"
        lines.append(header)
        lines.append("=" * len(header))

        # Main results
        significance = "SIGNIFICANT" if self.is_significant else "Not significant"
        lines.append(f"Result: {significance} (alpha = {self.alpha})")
        lines.append(f"Test statistic: {self.statistic:.4f}")
        lines.append(f"p-value: {format_p_value(self.p_value)}")

        if self.degrees_of_freedom is not None:
            if isinstance(self.degrees_of_freedom, tuple):
                df_str = f"({', '.join(map(str, self.degrees_of_freedom))})"
            else:
                df_str = str(self.degrees_of_freedom)
            lines.append(f"Degrees of freedom: {df_str}")

        # Effect size
        if self.effect_size is not None:
            lines.append(
                f"{self.effect_size_name}: {self.effect_size:.4f} ({self.effect_magnitude})"
            )

        # Confidence interval
        if self.confidence_interval is not None:
            ci_level = int((1 - self.alpha) * 100)
            ci_str = f"[{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]"
            lines.append(f"{ci_level}% Confidence Interval: {ci_str}")

        if verbose:
            # Sample sizes
            if self.sample_sizes is not None:
                if isinstance(self.sample_sizes, tuple):
                    lines.append(f"Sample sizes: {self.sample_sizes}")
                else:
                    lines.append(f"Sample size: {self.sample_sizes}")

            # Assumptions
            if self.assumptions_met:
                lines.append("\nAssumption Checks:")
                for assumption, met in self.assumptions_met.items():
                    status = "Met" if met else "Violated"
                    lines.append(f"  {assumption}: {status}")

            # Data summary
            if self.data_summary:
                lines.append("\nData Summary:")
                for key, value in self.data_summary.items():
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.4f}")
                    else:
                        lines.append(f"  {key}: {value}")

        # Interpretation
        if self.interpretation:
            lines.append(f"\nInterpretation:\n{self.interpretation}")

        # Routing fallback warning
        if getattr(self, "routing_source", "llm") == "fallback":
            lines.append(
                "\n⚠  Routed via regex fallback (confidence=60%). "
                "Verify the correct test was selected."
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        result_dict = {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "alternative": self.alternative,
        }

        if self.effect_size is not None:
            result_dict["effect_size"] = self.effect_size
            result_dict["effect_size_name"] = self.effect_size_name
            result_dict["effect_magnitude"] = self.effect_magnitude

        if self.confidence_interval is not None:
            result_dict["confidence_interval"] = self.confidence_interval

        if self.degrees_of_freedom is not None:
            result_dict["degrees_of_freedom"] = self.degrees_of_freedom

        if self.sample_sizes is not None:
            result_dict["sample_sizes"] = self.sample_sizes

        result_dict["assumptions_met"] = self.assumptions_met
        result_dict["data_summary"] = self.data_summary
        result_dict.update(self.extra_info)

        return result_dict

    def __str__(self) -> str:
        """String representation"""
        return self.summary()

    def __repr__(self) -> str:
        """Detailed representation"""
        return f"HypoResult(test='{self.test_name}', statistic={self.statistic:.4f}, p_value={self.p_value:.6f})"

    def plot(self, kind: str = "auto"):
        """
        Visualise this result.

        Delegates to :func:`hypotestx.explore.visualize.plot_result`.

        Parameters
        ----------
        kind : str
            ``"auto"`` (default), ``"p_value"``, ``"bar"``, ``"box"``.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ImportError
            If matplotlib is not installed.  Install with
            ``pip install matplotlib`` or ``pip install hypotestx[visualization]``.
        """
        from ..explore.visualize import plot_result

        return plot_result(self, kind=kind)
