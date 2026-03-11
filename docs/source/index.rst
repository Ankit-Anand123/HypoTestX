HypoTestX Documentation
========================

**Natural Language Hypothesis Testing — Powered by LLMs or Pure Regex**

HypoTestX lets you ask a plain-English question and a DataFrame and returns a
full structured ``HypoResult`` — statistic, p-value, effect size, confidence
interval, and human-readable interpretation.  Zero mandatory dependencies, pure
Python math core.

.. code-block:: python

   import hypotestx as hx
   import pandas as pd

   df = pd.read_csv("survey.csv")
   result = hx.analyze(df, "Do males earn more than females?")
   print(result.summary())

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation
   why_hypotestx

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   natural_language
   direct_api
   backends
   result_object
   assumption_checking
   power_analysis
   visualization
   reporting

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/analyze
   api/tests
   api/result
   api/backends
   api/power
   api/visualization

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog
   contributing
   license

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
