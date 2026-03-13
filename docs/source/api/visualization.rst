Visualization
=============

All visualization functions require **matplotlib** (install with
``pip install hypotestx[visualization]``).

.. autofunction:: hypotestx.explore.visualize.plot_result

.. autofunction:: hypotestx.explore.visualize.plot_distributions

.. autofunction:: hypotestx.explore.visualize.plot_p_value

Reporting Functions
-------------------

.. autofunction:: hypotestx.reporting.generator.text_report

.. autofunction:: hypotestx.reporting.generator.export_html

.. autofunction:: hypotestx.reporting.generator.export_pdf

.. autofunction:: hypotestx.reporting.generator.export_csv

.. autofunction:: hypotestx.reporting.generator.apa_report

Usage Examples
--------------

.. code-block:: python

   import hypotestx as hx

   result = hx.ttest_2samp(group1, group2)

   # Plot the result
   fig = result.plot()                       # auto
   fig = hx.plot_result(result, kind="bar")  # bar chart

   # Plot distributions
   fig = hx.plot_distributions(
       [group1, group2],
       labels=["Control", "Treatment"],
       kind="box",
   )

   # p-value visualization
   fig = hx.plot_p_value(
       p_value=result.p_value,
       alpha=result.alpha,
       test_statistic=result.statistic,
       alternative=result.alternative,
   )

   # Reports
   hx.generate_report(result, path="report.html", fmt="html")
   hx.generate_report(result, path="report.pdf",  fmt="pdf")   # needs weasyprint
