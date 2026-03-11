# Changelog

All notable changes to HypoTestX are documented here.  Versions follow
[Semantic Versioning](https://semver.org/).

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
