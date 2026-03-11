Statistical Test Functions
==========================

All test functions return a :class:`~hypotestx.core.result.HypoResult` object.

Parametric Tests
----------------

.. autofunction:: hypotestx.tests.parametric.one_sample_ttest

.. autofunction:: hypotestx.tests.parametric.two_sample_ttest

.. autofunction:: hypotestx.tests.parametric.paired_ttest

.. autofunction:: hypotestx.tests.parametric.anova_one_way

Non-parametric Tests
--------------------

.. autofunction:: hypotestx.tests.nonparametric.mann_whitney_u

.. autofunction:: hypotestx.tests.nonparametric.wilcoxon_signed_rank

.. autofunction:: hypotestx.tests.nonparametric.kruskal_wallis

Categorical Tests
-----------------

.. autofunction:: hypotestx.tests.categorical.chi_square_test

.. autofunction:: hypotestx.tests.categorical.fisher_exact_test

Correlation Tests
-----------------

.. autofunction:: hypotestx.tests.correlation.pearson_correlation

.. autofunction:: hypotestx.tests.correlation.spearman_correlation

.. autofunction:: hypotestx.tests.correlation.point_biserial_correlation

Public Aliases
--------------

The following short-form names are exposed at the top-level ``hypotestx``
namespace for convenience:

.. code-block:: python

   hx.ttest_1samp      # one_sample_ttest
   hx.ttest_2samp      # two_sample_ttest
   hx.welch_ttest      # two_sample_ttest with equal_var=False
   hx.ttest_paired     # paired_ttest
   hx.anova_1way       # anova_one_way
   hx.mannwhitney      # mann_whitney_u
   hx.wilcoxon         # wilcoxon_signed_rank
   hx.kruskal          # kruskal_wallis
   hx.chi2_test        # chi_square_test
   hx.fisher_exact     # fisher_exact_test
   hx.pearson          # pearson_correlation
   hx.spearman         # spearman_correlation
   hx.pointbiserial    # point_biserial_correlation
