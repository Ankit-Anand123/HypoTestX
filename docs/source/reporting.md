# Reporting

HypoTestX can produce APA-style text summaries, HTML reports with embedded
charts, PDF reports, and CSV exports of multiple results.

---

## `generate_report()`

The main reporting entry point. Produces HTML, PDF, or plain text.

```python
import hypotestx as hx

result = hx.ttest_2samp(group1, group2)

# Return as string
html = hx.generate_report(result, fmt="html")
text = hx.generate_report(result, fmt="text")

# Write directly to a file
hx.generate_report(result, path="report.html", fmt="html")
hx.generate_report(result, path="report.pdf",  fmt="pdf")    # see note below
hx.generate_report(result, path="report.txt",  fmt="text")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `result` | `HypoResult` | — | The result to report |
| `fmt` | `str` | `"text"` | Output format: `"html"`, `"pdf"`, or `"text"` |
| `path` | `str \| None` | `None` | File path to write; if `None`, returns the string |

---

## `export_html()`

Export a single result as a self-contained HTML file (with embedded chart if
matplotlib is installed):

```python
from hypotestx.reporting.generator import export_html

export_html(result, path="report.html")
```

The HTML report includes:
- Test name, statistic, p-value, effect size, confidence interval
- Human-readable interpretation
- Embedded matplotlib chart (if matplotlib ≥ 3.5 is installed)
- APA-formatted citation line

---

## `export_pdf()`

Convert the HTML report to a PDF using **WeasyPrint**.

```bash
pip install weasyprint
# or
pip install hypotestx[reporting]
```

```python
from hypotestx.reporting.generator import export_pdf

export_pdf(result, path="report.pdf")
```

> **Note:** WeasyPrint requires system libraries (Pango, Cairo, GDK-PixBuf) on
> Linux. On macOS and Windows, wheel packages are usually self-contained.

---

## `export_csv()`

Export one or more results as a CSV file — useful for building result tables
across many tests:

```python
from hypotestx.reporting.generator import export_csv

results = [
    hx.ttest_2samp(g1, g2),
    hx.pearson(x, y),
    hx.chi2_test(table),
]

export_csv(results, path="results.csv")
```

Each row is one test result with columns: `test_name`, `statistic`, `p_value`,
`is_significant`, `effect_size`, `effect_size_name`, `alpha`, `alternative`.

---

## `apa_report()`

Generate a single APA 7th-edition formatted string for a result:

```python
from hypotestx.reporting.formatters import apa_report

text = apa_report(result)
print(text)
# e.g.: "t(248) = 3.25, p = .001, d = 0.68, 95% CI [1.23, 4.56]"
```

---

## APA Formatting in `summary()`

`result.summary()` always produces an APA-compatible interpretation line at the
bottom:

```
Interpretation:
There is a statistically significant difference between the two groups
(t = 3.25, df = 248, p = 0.0012, Cohen's d = 0.68).
```

---

## Multiple Results Example

```python
import hypotestx as hx
from hypotestx.reporting.generator import export_csv, export_html

questions = [
    "Do males earn more than females?",
    "Is age correlated with salary?",
    "Are departments associated with performance tier?",
]

results = [hx.analyze(df, q, warn_fallback=False) for q in questions]

# CSV summary table
export_csv(results, path="all_results.csv")

# Individual HTML reports
for i, (q, r) in enumerate(zip(questions, results)):
    export_html(r, path=f"report_{i+1}.html")
```
