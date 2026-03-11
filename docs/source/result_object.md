# HypoResult Reference

Every statistical test — whether called via `analyze()` or a direct function —
returns a `HypoResult` object.  All fields are public attributes; none are
hidden behind properties except `is_significant` and `effect_magnitude`.

---

## Fields

### Core statistical output

| Field | Type | Description |
|---|---|---|
| `test_name` | `str` | Human-readable name, e.g. `"Welch's t-test (unequal variances)"` |
| `statistic` | `float` | Test statistic value (t, F, χ², U, W, r, …) |
| `p_value` | `float` | Two-sided or directional p-value |
| `effect_size` | `float \| None` | Effect size (Cohen's d, r, η², Cramér's V, …) |
| `effect_size_name` | `str \| None` | Name of the effect size measure used |
| `confidence_interval` | `tuple[float, float] \| None` | (lower, upper) confidence interval |
| `degrees_of_freedom` | `int \| float \| tuple \| None` | Degrees of freedom |
| `sample_sizes` | `int \| tuple \| None` | Per-group or total sample size(s) |
| `assumptions_met` | `dict[str, bool]` | Assumption check results (may be empty) |
| `interpretation` | `str \| None` | Plain-English interpretation |
| `data_summary` | `dict[str, Any]` | Descriptive stats (may be empty) |
| `alpha` | `float` | Significance level used |
| `alternative` | `str` | `"two-sided"`, `"greater"`, or `"less"` |

### Routing metadata

These fields are populated when the result was produced by `analyze()`. For
direct test calls they retain their defaults.

| Field | Type | Default | Description |
|---|---|---|---|
| `routing_confidence` | `float` | `1.0` | Routing confidence: `1.0` for an LLM, `0.6` for the regex fallback |
| `routing_source` | `str` | `"llm"` | Source of the routing decision: `"llm"` or `"fallback"` |

### Computed properties

| Property | Type | Description |
|---|---|---|
| `is_significant` | `bool` | `True` if `p_value < alpha` |
| `effect_magnitude` | `str` | Cohen's convention: `"negligible"`, `"small"`, `"medium"`, `"large"` — scale chosen automatically based on `effect_size_name` |

---

## `summary()` Output Format

```python
result.summary()                # default
result.summary(verbose=True)    # includes sample sizes, assumption checks, data summary
```

**Default output:**

```
[ Welch's t-test (unequal variances) ]
=======================================
Result: SIGNIFICANT (alpha = 0.05)
Test statistic: 3.2456
p-value: 0.0012
Degrees of freedom: 248.0
Cohen's d: 0.6834 (medium)
95% Confidence Interval: [1.2300, 4.5600]

Interpretation:
There is a statistically significant difference between the two groups
(t = 3.25, df = 248, p = 0.0012, Cohen's d = 0.68).
```

**When `routing_source == "fallback"`**, the summary appends:

```
⚠  Routed via regex fallback (confidence=60%). Verify the correct test was selected.
```

**Verbose output** additionally prints:

```
Sample sizes: (125, 125)

Assumption Checks:
  Normality (group1): Met
  Normality (group2): Met

Data Summary:
  group1_mean: 52500.1234
  group2_mean: 48200.5678
  pooled_std: 6200.4321
```

---

## `to_dict()`

Returns all fields as a plain Python dictionary — convenient for serialisation,
logging, or building DataFrames of results:

```python
d = result.to_dict()
# {
#   "test_name": "Welch's t-test (unequal variances)",
#   "statistic": 3.2456,
#   "p_value": 0.0012,
#   "is_significant": True,
#   "alpha": 0.05,
#   "alternative": "two-sided",
#   "effect_size": 0.6834,
#   "effect_size_name": "Cohen's d",
#   "effect_magnitude": "medium",
#   "confidence_interval": (1.23, 4.56),
#   "degrees_of_freedom": 248.0,
#   "sample_sizes": (125, 125),
#   "assumptions_met": {},
#   "data_summary": {},
# }
```

---

## `plot()`

Produce a matplotlib figure for the result.  Requires `matplotlib` (install with
`pip install hypotestx[visualization]`).

```python
fig = result.plot()                  # auto-selects chart type
fig = result.plot(kind="bar")        # grouped bar chart
fig = result.plot(kind="p_value")    # p-value on null distribution
fig = result.plot(kind="box")        # box plot of groups
fig.savefig("result.png")
```

| `kind` value | Description |
|---|---|
| `"auto"` | Best chart for the test type (default) |
| `"bar"` | Mean ± CI bar chart for two-group comparisons |
| `"box"` | Box plot of group distributions |
| `"p_value"` | p-value highlighted on the null distribution curve |

See [Visualization](visualization.md) for the full plotting guide.

---

## Accessing Individual Fields

```python
result = hx.analyze(df, "Do males earn more than females?")

print(result.test_name)            # "Welch's t-test (unequal variances)"
print(result.statistic)            # 3.2456
print(result.p_value)              # 0.0012
print(result.is_significant)       # True
print(result.effect_size)          # 0.6834
print(result.effect_size_name)     # "Cohen's d"
print(result.effect_magnitude)     # "medium"
print(result.confidence_interval)  # (1.23, 4.56)
print(result.alpha)                # 0.05
print(result.alternative)          # "two-sided"
print(result.routing_confidence)   # 0.6 (fallback) or 1.0 (LLM)
print(result.routing_source)       # "fallback" or "llm"
```
