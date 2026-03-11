HypoResult
==========

.. autoclass:: hypotestx.core.result.HypoResult
   :members:
   :show-inheritance:

Fields Reference
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``test_name``
     - ``str``
     - Human-readable test name
   * - ``statistic``
     - ``float``
     - Test statistic value (t, F, χ², U, W, r, …)
   * - ``p_value``
     - ``float``
     - p-value
   * - ``effect_size``
     - ``float | None``
     - Effect size (Cohen's d, r, η², Cramér's V, …)
   * - ``effect_size_name``
     - ``str | None``
     - Name of the effect size measure
   * - ``confidence_interval``
     - ``tuple[float, float] | None``
     - (lower, upper) confidence interval
   * - ``degrees_of_freedom``
     - ``int | float | None``
     - Degrees of freedom
   * - ``sample_sizes``
     - ``int | tuple | None``
     - Per-group or total sample size(s)
   * - ``assumptions_met``
     - ``dict[str, bool]``
     - Assumption check results
   * - ``interpretation``
     - ``str | None``
     - Plain-English interpretation
   * - ``alpha``
     - ``float``
     - Significance level used
   * - ``alternative``
     - ``str``
     - ``"two-sided"``, ``"greater"``, or ``"less"``
   * - ``routing_confidence``
     - ``float``
     - Routing confidence: ``1.0`` (LLM) or ``0.6`` (fallback)
   * - ``routing_source``
     - ``str``
     - ``"llm"`` or ``"fallback"``
   * - ``is_significant``
     - ``bool`` *(property)*
     - ``True`` if ``p_value < alpha``
   * - ``effect_magnitude``
     - ``str`` *(property)*
     - ``"negligible"``, ``"small"``, ``"medium"``, or ``"large"``
