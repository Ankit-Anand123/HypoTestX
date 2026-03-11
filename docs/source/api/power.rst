Power Analysis
==============

.. autofunction:: hypotestx.power.sample_size.n_ttest_two_sample

.. autofunction:: hypotestx.power.analysis.power_ttest_two_sample

Functions Reference
-------------------

``n_ttest_two_sample``
~~~~~~~~~~~~~~~~~~~~~~

Calculate the required sample size per group to detect a given effect size at
the desired power level for an independent two-sample t-test.

.. code-block:: python

   import hypotestx as hx

   # Medium effect, standard alpha, 80% power
   n = hx.n_ttest_two_sample(effect_size=0.5, alpha=0.05, power=0.8)
   print(f"Required n per group: {n}")   # 64

   # Small effect, 90% power
   n = hx.n_ttest_two_sample(effect_size=0.2, alpha=0.05, power=0.9)

``power_ttest_two_sample``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the achieved power for a two-sample t-test given the observed effect
size and sample sizes.

.. code-block:: python

   pow_result = hx.power_ttest_two_sample(
       effect_size=0.4,
       n1=30,
       n2=30,
       alpha=0.05,
   )
   print(f"Achieved power: {pow_result.power:.2f}")
