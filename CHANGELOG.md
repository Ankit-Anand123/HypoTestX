# Changelog

All notable changes to HypoTestX are documented here.  Versions follow
[Semantic Versioning](https://semver.org/).

---

## v1.0.6 — Released (current)

Routing transparency and documentation release.

- **`warn_fallback` parameter** — `analyze()` now accepts `warn_fallback=True`
  (default). Emits a `UserWarning` when the built-in regex fallback is used,
  directing users to a real LLM backend for better accuracy. Suppress with
  `warn_fallback=False`.
- **`routing_confidence` field** — `HypoResult` now exposes `routing_confidence`
  (float 0–1; `1.0` for LLM, `0.6` for fallback) and `routing_source`
  (`"llm"` or `"fallback"`).
- **Fallback warning in `summary()`** — when `routing_source == "fallback"`,
  `result.summary()` appends a ⚠ notice advising the user to verify the
  selected test.
- **`routing_source` field on `RoutingResult`** — the internal dataclass now
  carries `routing_source` alongside `confidence`.
- **ReadTheDocs documentation** — full Sphinx/furo documentation site with
  quickstart, user guide, API reference, and contributing guide.
  Published at https://hypotestx.readthedocs.io
- **`pyproject.toml` docs extras** updated to `sphinx>=7.0`, `furo>=2024.1.29`,
  `myst-parser>=2.0`.
- README: fixed roadmap versioning (linear v0.1.0 → v0.2.0 → v1.0.0 sequence);
  added "Why HypoTestX?" comparison table; added `routing_confidence` and
  `routing_source` to the HypoResult reference.

---

## v1.0.0 — Released

Production-ready release with domain packages, reporting, and visualization.

- **Domain-specific packages**: `hypotestx.domains` — clinical/medical,
  A/B testing, finance, survey analysis
- **HTML / PDF reporting**: `generate_report()`, `export_html()`, `export_pdf()`
  (PDF requires `weasyprint`)
- **Visualization**: `result.plot()`, `plot_result()`, `plot_distributions()`,
  `plot_p_value()` — requires optional `matplotlib`
- **Azure OpenAI backend**: `backend="azure"` with correct deployment URL,
  `api-key` header, and `api_version` parameter
- **Routing confidence warnings**: `routing_confidence` and `routing_source`
  fields on `HypoResult`; `UserWarning` emitted when fallback is used
- **Routing validation**: explicit column checks per test type with actionable
  error messages before dispatch
- **Structured logging**: `logging.getLogger("hypotestx")` throughout
- **Duck-typed backends**: any object exposing `.route()` accepted by `get_backend()`
- **Division-by-zero fix**: Welch/Student t-tests guard against zero-variance groups
- 532 tests passing

---

## v0.2.0 — Released

LLM-powered natural-language interface.

- **`analyze()` function** — single entry-point natural-language interface
- **Plug-in LLM backend system** — swap any backend with one keyword arg
  - `FallbackBackend` — built-in regex router (zero deps, offline, always works)
  - `OllamaBackend` — local Ollama (free, offline)
  - `GeminiBackend` — Google Gemini free tier (1 500 req/day)
  - `OpenAICompatBackend` — OpenAI, Groq, Together AI, Mistral, Perplexity
  - `HuggingFaceBackend` — HuggingFace Inference API + local `transformers`
  - `CallableBackend` — wraps any `callable(messages) -> str`
  - `LLMBackend` ABC — subclass to integrate any custom LLM
- `RoutingResult` structured dataclass for intent extraction
- `SchemaInfo` DataFrame summary passed to LLMs as context
- Regex fallback pattern table covering all test types
- 483 tests passing

---

## v0.1.0 — Released

Initial public release.

- Complete parametric test suite: one-sample t-test, two-sample t-test,
  Welch's t-test, paired t-test, one-way ANOVA
- Non-parametric tests: Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis
- Categorical tests: chi-square (independence + goodness-of-fit), Fisher's exact
- Correlation tests: Pearson, Spearman, point-biserial
- Pure Python math core: distributions, special functions, numerical integration —
  zero mandatory dependencies
- Assumption checking: Shapiro-Wilk, Levene, Bartlett, Jarque-Bera
- Power analysis: sample size calculation, post-hoc power for two-sample t-tests
- Bootstrap and permutation tests
- APA-style reporting via `apa_report()`
- Structured `HypoResult` with statistic, p-value, effect size, CI, interpretation

---

## v1.1.0 — Planned

- Two-way ANOVA and repeated-measures ANOVA
- Regression-based tests (linear, logistic)
- Automatic assumption-driven test selection
- Streaming LLM responses for verbose mode
- `analyze()` result explains *why* a test was chosen

---

## v1.2.0 — Planned

- Bayesian alternatives (Bayesian t-test, Bayes factor)
- Time series stationarity and change-point tests
- Meta-analysis tools
- Interactive Jupyter widgets for results
