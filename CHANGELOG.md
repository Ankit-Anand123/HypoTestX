# Changelog

All notable changes to HypoTestX are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.5] — 2026-03-10

### Added
- **Visualization** — `visualize.py` fully implemented (was a stub):
  - `plot_result(result, kind)` — auto-selects bar or p-value curve
  - `plot_distributions(groups, labels, kind)` — box / bar / violin charts
  - `plot_p_value(p_value, alpha, ...)` — null distribution with rejection regions
  - `generate_report(result, path, fmt)` — HTML, text, and PDF (weasyprint) reports
  - `result.plot()` delegation method on `HypoResult`
  - All helpers gracefully raise `ImportError` with install hint when `matplotlib` is absent
- **Azure OpenAI backend** — `OpenAICompatBackend` now fully supports Azure:
  - Auto-detected via `provider="azure"` or by URL containing `.openai.azure.com`
  - Correct Azure URL: `{base_url}/openai/deployments/{model}/chat/completions?api-version=...`
  - Uses `api-key` header (not `Authorization: Bearer ...`)
  - `"azure"` shorthand added to `get_backend()` factory
  - `api_version` parameter (default `"2024-02-01"`)
- **`backend_options` passthrough** — `analyze()` now accepts a `backend_options` dict
  for provider-specific kwargs not covered by the standard whitelist
- **`export_html()` and `export_pdf()`** — new exporters in `reporting/generator.py`
  delegating to `generate_report()`; both re-exported from the top-level `hypotestx` namespace
- **Routing validation** — `_validate_routing_columns()` checks required column fields
  per test type and raises a descriptive `ValueError` with actionable guidance before dispatch
- **Structured logging** — `logging.getLogger("hypotestx")` used throughout `engine.py`;
  set to `DEBUG` level to trace routing and backend calls; silent by default
- **Duck-typed backend acceptance** — `get_backend()` now accepts any object with a callable
  `.route()` method, enabling lightweight stubs and third-party wrappers without formal
  `LLMBackend` inheritance
- **`reporting` and `all` optional extras** in `pyproject.toml`:
  - `reporting` — `weasyprint>=53.0` for PDF export
  - `all` — convenience extra that pulls in all optional deps
- **Expanded test suite** — 532 tests passing (up from 483); 1 intentional skip:
  - `tests/test_reporting/test_visualize.py` — 17 visualization tests
    (12 are `skipUnless(matplotlib)`, all pass when matplotlib is installed)
  - `tests/test_core/test_azure_backend.py` — 17 Azure backend tests
  - Edge-case tests added to `test_parametric.py`: zero-variance, empty, single-element groups
- **Security & API Key Best Practices** section added to README

### Fixed
- **Welch's t-test / Student's t-test** — division-by-zero when one or both groups have zero
  variance (constant data); raises `ValueError` with a clear diagnostic message
- **Azure backend** — previously used `Authorization: Bearer ...` header and the standard
  `/chat/completions` endpoint; now uses the correct Azure deployment URL and `api-key` header
- **README** — visualization section corrected to show real implemented APIs; removed
  references to unimplemented stubs (`plot_effect_size`, `plot_assumptions`)

---

## [1.0.0] — 2026-03-09

### Added
- **197 new tests** — validation, inference, reporting, testsuite, bootstrap, descriptive extras
  (total test suite: 483 passing, 3 skipped)
- `utils/validation.py` — `validate_dataframe`, `validate_columns`, `validate_numeric_column`,
  `validate_categorical_column`, `validate_sample_size`, `validate_alpha`,
  `validate_probability`, `validate_alternative`
- `stats/inference.py` — `confidence_interval_mean`, `confidence_interval_proportion`
  (Wilson & normal), `confidence_interval_difference_of_means`, `z_test_one_sample`
- `stats/distributions.py` — re-exports `Normal`, `StudentT`, `ChiSquare`, `F` for convenience
- `reporting/templates.py` — `APA_TEMPLATES`, `render_apa`, `render_plain`, `render_one_line`
- `core/testsuite.py` — `TestSuite` class for running and aggregating multiple tests
- `tests/regression.py` — `LinearRegressionTest` (OLS) with R-squared and F-statistic
- LLM engine (`core/llm/`) — `analyze()` natural-language dispatch, FallbackBackend (regex),
  OllamaBackend, OpenAICompatBackend, GeminiBackend, HuggingFaceBackend

### Fixed
- `RoutingResult.alpha` defaulted to `0.05`, overriding caller-supplied `alpha` in `analyze()`
- `Normal.ppf(0.5)` returned `0.0` for non-standard normals; now returns `self.mu`
- `FallbackBackend` applied directional `alternative` for two-sample tests where group
  ordering is alphabetical, causing `p_value ≈ 1.0` for "greater" questions
- `point_biserial_correlation` via engine forced `float()` on the binary column, crashing
  on string-coded groups
- Paired t-test test data had constant differences (std=0), causing division by zero

### Changed
- Version bumped to `1.0.0`
- `pyproject.toml` classifier updated to `5 - Production/Stable`
- Author corrected to `HypoTestX Contributors`

### Removed
- `setup.py` — superseded by `pyproject.toml`

---

## [0.1.0] — 2026-03-09

### Added

#### Core statistical tests (pure Python, zero dependencies)
- `one_sample_ttest` — one-sample Student's t-test
- `two_sample_ttest` — two-sample Student's / Welch's t-test
- `paired_ttest` — paired-samples t-test
- `anova_one_way` — one-way ANOVA with eta-squared effect size
- `mann_whitney_u` — Mann-Whitney U test with rank-biserial r
- `wilcoxon_signed_rank` — Wilcoxon signed-rank test (one-sample and paired)
- `kruskal_wallis` — Kruskal-Wallis H test
- `chi_square_test` — chi-square test of independence and goodness-of-fit
- `fisher_exact_test` — Fisher's exact test for 2×2 tables
- `pearson_correlation` — Pearson r with t-test significance
- `spearman_correlation` — Spearman rank correlation
- `point_biserial_correlation` — point-biserial correlation

#### Pure Python math core
- Normal, Student-t, Chi-square, F distributions (PDF, CDF, PPF)
- `mean`, `std`, `variance`, `correlation` (no NumPy)
- Linear algebra utilities (matrix operations, determinant, inverse)
- Special functions: gamma, beta, erf, incomplete beta / gamma

#### Power analysis & sample size
- `power_ttest_one_sample`, `power_ttest_two_sample`, `power_ttest_paired`
- `power_anova`, `power_chi_square`, `power_correlation`
- Matching `n_*` sample-size calculators for all the above

#### Assumption checks
- `shapiro_wilk` — Shapiro-Wilk normality test
- `levene_test` — Levene's test of equal variances
- `bartlett_test` — Bartlett's test
- `jarque_bera` — Jarque-Bera normality test
- `check_normality`, `check_equal_variances` — convenience wrappers

#### Descriptive statistics & bootstrap
- `DescriptiveStats`, `describe`, `five_number_summary`
- `detect_outliers`, `frequency_table`, `compare_groups`
- `bootstrap_ci`, `bootstrap_mean_ci`, `bootstrap_two_sample_ci`
- `bootstrap_test`, `permutation_test`

#### LLM-powered natural language interface — `analyze()`
- Top-level `hx.analyze(df, question, backend=..., alpha=0.05, verbose=False)`
- Full plug-in backend architecture (`LLMBackend` ABC + `CallableBackend`)
- **FallbackBackend** — regex router, zero dependencies, works offline (default)
- **OllamaBackend** — local Ollama server (llama3.2, phi4, gemma2, mistral, …)
- **GeminiBackend** — Google Gemini REST API (free tier: 1500 req/day)
- **OpenAICompatBackend** — OpenAI, Groq, Together AI, Perplexity, Mistral, Azure
- **HuggingFaceBackend** — HuggingFace Inference API and local `transformers`
- `get_backend()` factory resolves strings / instances / callables
- `build_schema()` creates a `SchemaInfo` snapshot from pandas or polars DataFrames
- All HTTP calls use `urllib.request` (stdlib only — no `requests` required)

#### Reporting
- `apa_report` — APA 7th edition formatted result strings
- `text_report` — verbose human-readable report
- `batch_report` — report multiple results at once
- `export_csv` — export results to CSV
- `format_p`, `format_ci`, `format_effect`, `apa_stat` — formatting helpers

#### CLI (`hypotestx` command)
- `hypotestx analyze <file.csv> "<question>"` — run analysis from the shell
- `--backend`, `--api-key`, `--model`, `--host` — backend options
- `--alpha`, `--verbose`, `--format` (summary / json / apa)
- `hypotestx backends` — list available backends
- `hypotestx version` — show installed version

#### Packaging
- Pure Python, no mandatory dependencies
- `pyproject.toml` with optional extras: `dev`, `docs`, `visualization`
- Python 3.8 – 3.13 supported

---

## [Unreleased]

### Planned for 0.2.0
- Two-way ANOVA and repeated-measures ANOVA
- Simple and multiple linear regression tests
- Automatic assumption-driven test selection in `analyze()`
- Streaming LLM response support

### Planned for 0.3.0
- Bayesian t-test and Bayes factor
- Time-series stationarity (ADF, KPSS) and change-point detection
- Meta-analysis (fixed-effect, random-effect models)

### Planned for 1.0.0
- Domain packages: A/B testing, clinical trials, finance
- Comprehensive Sphinx documentation
- Publication-ready PDF / HTML reporting
