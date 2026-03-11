# Natural Language Interface

`hx.analyze()` is the main entry point for the natural-language interface. Pass
a plain-English question and a DataFrame — HypoTestX figures out the right test,
extracts the right columns, and returns a full `HypoResult`.

## Signature

```python
hx.analyze(
    df,
    question,
    backend=None,
    alpha=0.05,
    verbose=False,
    warn_fallback=True,
    **kwargs,
)
```

## Supported Question Patterns

### Two-group comparisons

The most common pattern — compare a numeric variable across two categorical groups:

```python
hx.analyze(df, "Do males earn more than females?")
hx.analyze(df, "Is there a difference between group A and group B?")
hx.analyze(df, "Are premium customers different from basic customers?")
hx.analyze(df, "Test whether method 1 is better than method 2")
hx.analyze(df, "Is the treatment group higher than the control group?")
```

### One-sample tests

Test whether a population mean equals a specific value:

```python
hx.analyze(df, "Is the average score different from 75?")
hx.analyze(df, "Test if the mean equals 50")
hx.analyze(df, "Is the average significantly greater than 100?")
hx.analyze(df, "Does the typical response time equal 200ms?")
```

### Correlation / relationships

Test linear or monotonic associations between two numeric variables:

```python
hx.analyze(df, "Is age correlated with salary?")
hx.analyze(df, "Is salary related to years of experience?")
hx.analyze(df, "Does age predict salary?")
hx.analyze(df, "Is there a linear relationship between height and weight?")
```

### Categorical associations

Test independence between two categorical variables:

```python
hx.analyze(df, "Is there an association between gender and department?")
hx.analyze(df, "Are treatment outcome and gender independent?")
hx.analyze(df, "Are product preference and region related?")
hx.analyze(df, "Is product choice associated with customer type?")
```

### Multi-group comparisons

Compare a numeric variable across three or more groups (ANOVA / Kruskal-Wallis):

```python
hx.analyze(df, "Compare satisfaction scores across all regions")
hx.analyze(df, "Are there differences in performance across three teams?")
hx.analyze(df, "Do all departments have the same average salary?")
hx.analyze(df, "Is there an effect of treatment across multiple groups?")
```

### Paired / before-after

Compare two measurements from the same subjects:

```python
hx.analyze(df, "Did scores improve from pre_score to post_score?")
hx.analyze(df, "Compare before and after treatment")
hx.analyze(df, "Is there a change in weight from baseline to follow-up?")
hx.analyze(df, "Do repeated measurements differ?")
```

---

## How the Regex Fallback Works

When `backend=None` (the default), HypoTestX uses its built-in `FallbackBackend`:

1. **Test selection** — the question is matched against a priority-ordered table of
   regex patterns. The first matching pattern wins.
2. **Direction detection** — keywords like "higher", "greater", "more" map to
   `alternative="greater"`; "lower", "fewer", "less" map to `alternative="less"`.
3. **Column mapping** — schema column names that appear literally in the question are
   used. Unmatched columns fall back to the first numeric / first categorical in the
   schema.
4. **`mu` extraction** — for one-sample tests, the first number in the question is
   used as the null hypothesis value.

### Fallback Limitations

- **Literal column matching only** — if the question says "income" but the column is
  named `annual_salary`, the mapping will fail.
- **No semantic understanding** — "compare apples and oranges" may mismatch if
  `apples` and `oranges` are not column names.
- **Routing confidence is 0.6** — always verify the selected test makes sense.

When the fallback is used, HypoTestX emits a `UserWarning`:

```
UserWarning:
[HypoTestX] Using built-in regex fallback to route: "Do males earn more than females?"
  Confidence is limited (~0.6). For better accuracy use a real LLM backend:
    hx.analyze(df, question, backend="gemini", api_key="...")
    hx.analyze(df, question, backend="ollama")  # free, offline
  Suppress this with: warn_fallback=False
```

To suppress the warning:

```python
result = hx.analyze(df, "Do males earn more?", warn_fallback=False)
```

---

## Using LLM Backends for Better Routing

A real LLM backend understands semantic meaning, renames, and complex phrasing:

```python
import os
import hypotestx as hx

# Gemini (free tier)
result = hx.analyze(
    df,
    "Is annual income statistically different between the two genders?",
    backend="gemini",
    api_key=os.environ["GEMINI_API_KEY"],
    model="gemini-2.0-flash",
    temperature=0.0,
)

# Ollama (offline, free)
result = hx.analyze(
    df,
    "Do the three product categories have the same average rating?",
    backend="ollama",
    model="llama3.2",
)
```

See [Backends](backends.md) for the full list of supported backends.

---

## The `routing_confidence` Field

After `analyze()` returns, you can inspect how confident HypoTestX was about the
test selection:

```python
result = hx.analyze(df, "Is salary correlated with age?")

print(result.routing_confidence)   # 0.6 (fallback) or 1.0 (LLM)
print(result.routing_source)       # 'fallback' or 'llm'
```

When `routing_source == "fallback"`, `result.summary()` also appends a warning:

```
⚠  Routed via regex fallback (confidence=60%). Verify the correct test was selected.
```

---

## verbose=True Output

Pass `verbose=True` to see exactly what HypoTestX is doing:

```python
result = hx.analyze(
    df,
    "Is salary different between genders?",
    backend="gemini",
    api_key="AIza...",
    model="gemini-2.0-flash",
    verbose=True,
)
```

Output:

```
[HypoTestX] Schema: 500 rows, columns: ['gender', 'salary', 'age', 'dept']
[HypoTestX] Backend: GeminiBackend
[HypoTestX] Question: 'Is salary different between genders?'
[HypoTestX] Routing confidence: 100% (source: llm)
[HypoTestX] Routing -> test='two_sample_ttest', confidence=1.0
[HypoTestX] Reasoning: Two distinct groups (male/female) compared on a numeric salary column
```
