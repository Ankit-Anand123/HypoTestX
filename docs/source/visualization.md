# Visualization

HypoTestX provides several helpers for plotting test results and distributions.
All visualization functions require **matplotlib** (and some support the optional
plotly backend):

```bash
pip install matplotlib
# or, for matplotlib + plotly together:
pip install hypotestx[visualization]
```

---

## `result.plot()`

Every `HypoResult` has a `.plot()` method that auto-selects the best chart type
for the test:

```python
import hypotestx as hx

result = hx.ttest_2samp(group1, group2)
fig = result.plot()           # auto
fig = result.plot(kind="bar")          # grouped bar with CI
fig = result.plot(kind="box")          # box plot
fig = result.plot(kind="p_value")      # p-value on null distribution
fig.savefig("result.png")
```

| `kind` | Description |
|---|---|
| `"auto"` | Best chart type for the test (default) |
| `"bar"` | Mean ± 95 % CI bar chart for two-group tests |
| `"box"` | Box plots of group distributions |
| `"p_value"` | Test statistic highlighted on the null distribution |

---

## `plot_result()`

Standalone function — equivalent to `result.plot()`:

```python
from hypotestx.explore.visualize import plot_result
# or via public API (if exposed in __init__.py)
import hypotestx as hx

fig = hx.plot_result(result, kind="auto")
fig.show()
```

---

## `plot_distributions()`

Plot the raw distributions of one or more groups:

```python
import hypotestx as hx

fig = hx.plot_distributions(
    [group1, group2],
    labels=["Control", "Treatment"],
    kind="box",            # "box" (default) | "bar" | "violin"
    title="Group Comparison",
)
fig.show()
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `groups` | `list[list[float]]` | — | Data for each group |
| `labels` | `list[str] \| None` | `None` | Group labels |
| `kind` | `str` | `"box"` | Chart type: `"box"`, `"bar"`, `"violin"` |
| `title` | `str \| None` | `None` | Plot title |

---

## `plot_p_value()`

Visualise a p-value on the null distribution for a given test statistic:

```python
import hypotestx as hx

fig = hx.plot_p_value(
    p_value=0.023,
    alpha=0.05,
    test_statistic=2.41,
    alternative="two-sided",
)
fig.show()
```

The rejection region is highlighted in red; the critical value lines and
the test statistic are annotated.

---

## `generate_report()`

Generate a formatted report — HTML, PDF, or plain text:

```python
import hypotestx as hx

result = hx.ttest_2samp(group1, group2)

# Return as string
html = hx.generate_report(result, fmt="html")
text = hx.generate_report(result, fmt="text")

# Write to file
hx.generate_report(result, path="report.html", fmt="html")
hx.generate_report(result, path="report.pdf",  fmt="pdf")   # requires weasyprint
hx.generate_report(result, path="report.txt",  fmt="text")
```

| `fmt` | Description | Extra requirement |
|---|---|---|
| `"html"` | HTML with embedded chart (if matplotlib is installed) | None |
| `"pdf"` | PDF via WeasyPrint | `pip install weasyprint` |
| `"text"` | Plain text summary | None |

---

## Reporting module helpers

For lower-level control:

```python
from hypotestx.reporting.generator import export_html, export_pdf, export_csv

# Single result to HTML file
export_html(result, path="report.html")

# PDF (requires weasyprint)
export_pdf(result, path="report.pdf")

# Multiple results to CSV
export_csv([result1, result2, result3], path="results.csv")
```

---

## Notes

- All `plot_*` functions return a `matplotlib.figure.Figure` object. Call
  `.show()` to display interactively or `.savefig(path)` to write to disk.
- If matplotlib is not installed, these functions raise an `ImportError` with
  a helpful install message.
- `plot_effect_size()`, `plot_assumptions()`, and `generate_apa_report()` from
  older documentation are not yet implemented. Use `result.plot()`,
  `plot_distributions()`, and `apa_report()` respectively.
