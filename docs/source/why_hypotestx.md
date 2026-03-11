# Why HypoTestX?

## The Problem

Statistical hypothesis testing is a core skill in data science, research, and
engineering. Yet every existing option forces a painful trade-off:

### Option A — scipy (and friends)

scipy is excellent, but to use it you must:

1. **Already know which test to run.** Is this a t-test or a Mann-Whitney? Should
   I use Welch's correction? Is this paired or independent?
2. **Manually prepare your data.** Slice the DataFrame yourself, convert to arrays,
   handle NaN values, split groups by label.
3. **Interpret raw numbers.** `scipy.stats.ttest_ind()` returns a tuple
   `(statistic, p_value)`. No effect size, no confidence interval, no interpretation.

For a seasoned statistician this is fine. For everyone else — the analyst, the ML
engineer, the researcher who is not a statistician by training — it creates a
significant barrier.

### Option B — Ask a Chat LLM

Modern LLMs can answer "do males earn more than females in my data?" surprisingly
well — in natural language. But the answer:

- **Cannot be embedded in code.** It's a paragraph of text, not a structured object.
- **Is not reproducible.** Run the same question again and you may get a different answer.
- **Has no audit trail.** You cannot verify which test it ran or with which parameters.
- **Cannot be composed.** You can't call `.p_value`, `.effect_size`, or `.plot()` on a chat reply.

### The Gap HypoTestX Fills

| Feature | scipy | Ask an LLM | HypoTestX |
|---|---|---|---|
| Natural language input | ❌ | ✅ | ✅ |
| Structured result object | ❌ | ❌ | ✅ |
| Effect size + CI included | Manual | ❌ | ✅ |
| Reproducible / embeddable | ✅ | ❌ | ✅ |
| Works offline | ✅ | ❌ | ✅ (fallback) |
| Auto test selection | ❌ | ❌ | ✅ |

HypoTestX gives you:

- A **plain-English interface** so you don't need to know which test to pick
- A **structured `HypoResult`** with every number you need, every time
- **Effect sizes and confidence intervals** included automatically
- **Reproducible, embeddable results** you can version-control and audit
- A **regex fallback** that works completely offline — no API key, no network
- An **LLM backend system** where you swap in any model with a single keyword

---

## When to Use the Direct API Instead

HypoTestX also exposes all 12 test functions directly. Use these when:

- You already know exactly which test you need
- You want the most explicit, readable code possible
- You're writing a library or module that others will read

```python
# Very explicit — no routing, no ambiguity
result = hx.ttest_2samp(
    group1, group2,
    alternative="greater",
    equal_var=False,   # Welch's correction
    alpha=0.01,
)
```

The natural-language interface and the direct API return identical `HypoResult`
objects — the same `.p_value`, `.effect_size`, `.summary()`, and `.plot()` work
on both.
