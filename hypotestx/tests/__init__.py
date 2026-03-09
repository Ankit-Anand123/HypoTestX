"""
hypotestx.tests — Statistical test functions

Parametric
----------
one_sample_ttest, two_sample_ttest, paired_ttest, anova_one_way

Non-parametric
--------------
mann_whitney_u, wilcoxon_signed_rank, kruskal_wallis

Categorical
-----------
chi_square_test, fisher_exact_test

Correlation
-----------
pearson_correlation, spearman_correlation, point_biserial_correlation
"""

from .parametric import (
    one_sample_ttest,
    two_sample_ttest,
    paired_ttest,
    anova_one_way,
)
from .nonparametric import (
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
)
from .categorical import (
    chi_square_test,
    fisher_exact_test,
)
from .correlation import (
    pearson_correlation,
    spearman_correlation,
    point_biserial_correlation,
)

__all__ = [
    # Parametric
    "one_sample_ttest",
    "two_sample_ttest",
    "paired_ttest",
    "anova_one_way",
    # Non-parametric
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    # Categorical
    "chi_square_test",
    "fisher_exact_test",
    # Correlation
    "pearson_correlation",
    "spearman_correlation",
    "point_biserial_correlation",
]
