# polars_reg Roadmap

## 1. R Equivalence Layer (DONE)

Mirror the existing Stata equivalence system for R users.

### 1a. `to_r()` — Generate equivalent R code
- [x] Add `polars_reg/r_equiv.py` with `to_r()` function (same signature as `to_stata()`)
- [x] Translation mapping for all estimators (ols, iv2sls, liml, gmm_iv, panel_fe/re/fd)
- [x] Include library() calls in output (e.g., `library(fixest)`)
- [x] Export `to_r` from `__init__.py`

### 1b. `compare_r()` — Run both and compare (via rpy2)
- [x] Add `compare_r()` to `polars_reg/r_equiv.py`
- [x] Use rpy2 to execute R code and extract coefficients/SEs when available
- [x] Fall back gracefully when rpy2 is not installed (same pattern as pystata fallback)
- [x] Return `ComparisonReport`

### 1c. Update showcase notebook
- [x] Add Section 10: R Equivalence showing `to_r()` output for all estimator types
- [x] Add `compare_r()` example (with graceful fallback)

### 1d. Tests for R translation
- [x] `tests/test_r_equiv.py` — 25 unit tests for `to_r()` translation (no R needed)
- [x] Verify each estimator produces syntactically correct R code
- [x] Test edge cases: no intercept, multi-way cluster, FE + IV, indicators, interactions

---

## 2. Cross-Validated Parity Tests (Stata + R) (DONE)

Extend the existing `tests/stata_compare.py` infrastructure to also verify against R.

### 2a. R parity infrastructure
- [x] `tests/r_compare.py` mirroring `tests/stata_compare.py`
  - `r_available()`, `r_has_package()`, `to_r_script()`, `_run_r_script()`, `assert_r_parity()`
- [x] R scripts: read CSV data, run regression, write results CSV
- [x] R package handling: fixest (FE/clustering), AER::ivreg (LIML), plm (panel)

### 2b. Combined parity test file
- [x] `tests/test_cross_parity.py` — 30 tests (14 Stata + 16 R), full test matrix
- [x] `tests/test_stata_parity.py` — 15 translation tests + Stata parity tests

### 2c. Tolerance hierarchy
- [x] TIGHT (1e-6): OLS, 2SLS; REGHDFE (2e-5): FE absorption; MEDIUM (1e-4): GMM; LOOSE (2e-3): LIML; PANEL (5e-2): plm differences

---

## 3. GroupBy Regression (DONE)

Run the same regression per group and collect results.

### 3a. Core API
- [x] `groupby_reg(pr.ols, "y ~ x1 + x2", data=df, group_by="ticker")` returns `GroupRegressionResult`
- [x] Works with any estimator function (ols, iv2sls, liml, gmm_iv, panel_fe, etc.)
- [x] `GroupRegressionResult`: dict-like (`.keys()`, `[group]`, `.values()`, `.items()`)
- [x] `.coef_table()` — stacked Polars DataFrame with group column
- [x] `.summary()` — compact multi-group summary

### 3b. Integration with regtable
- [x] `regtable(*group_result.values())` works naturally

### 3c. Edge cases
- [x] Groups with too few observations → skip with warning (min_obs parameter)
- [x] Groups with singular X'X → skip with warning, don't crash
- [x] Failed groups tracked in `.failed` dict

---

## 4. Pandas Compatibility (DONE)

Accept `pandas.DataFrame` as input — convert to Polars internally, run as normal, and return Polars-native results.

### 4a. Input handling
- [x] `_utils.py: ensure_polars(data)` — centralized pandas→Polars conversion
- [x] Called in all estimators (ols, iv2sls, liml, gmm_iv, panel_fe/re/fd, panel_ab, probit, logit, quantreg)
- [x] Pandas import only inside isinstance check — remains optional dependency

### 4b. Tests
- [x] `tests/test_pandas_compat.py` — 11 tests verifying identical results for all estimators

---

## 5. Other Feature Gaps

### 5a. Interaction terms in formula (DONE)
- [x] Support `x1:x2` syntax for interaction terms in formula parser
- [x] Support `x1*x2` (full factorial: main effects + interaction)
- [x] Three-way and higher-order interactions
- [x] Stata equivalence: `c.x1#c.x2` syntax
- [x] R equivalence: `x1:x2` syntax

### 5b. HAC / Driscoll-Kraay standard errors (DONE)
- [x] Newey-West (HAC) for time series
- [x] Driscoll-Kraay for panel data with cross-sectional dependence

### 5c. Additional diagnostics (DONE)
- [x] Wald test for linear restrictions
- [x] Hausman test (FE vs RE)
- [x] Weak instrument diagnostics (Stock-Yogo critical values, Kleibergen-Paap)

---

## 6. Polish & Packaging

### 6a. Documentation (DONE)
- [x] README with quickstart, installation, feature overview
- [x] API reference via pdoc (auto-generated from docstrings): `uv run pdoc polars_reg -o docs/api --docformat google`
- [x] Add showcase notebook link to README

### 6b. CI/CD (DONE)
- [x] GitHub Actions: run tests on push (Python 3.11, 3.12)
- [x] Lint (ruff check) + format check (ruff format --check) in CI
- [x] Optional R parity job (manual trigger or [r-parity] commit, installs fixest/AER/plm/sandwich/lmtest)
- [x] Auto-publish to PyPI on tagged release (pypa/gh-action-pypi-publish)

### 6c. Performance
- [x] Benchmark suite (N = 1K, 10K, 100K, 1M)
- [x] Sparse FE dummies for large group counts (>1000) — handled by bincount-based demeaning
- [x] Profile demeaning for bottlenecks
- [x] Vectorize _clustered_meat, _is_nested, drop_singletons (91x speedup)

---

## 7. regtable Export Formats (DONE)

### 7a. LaTeX export
- [x] `regtable(..., format="latex")` returns LaTeX tabular string
- [x] Proper escaping of underscores, special characters
- [x] Booktabs style (toprule/midrule/bottomrule)
- [x] Significance stars as superscripts

### 7b. HTML export
- [x] `regtable(..., format="html")` returns HTML table string
- [x] CSS classes for styling (coefficient, se, header, footer)
- [x] Jupyter notebook auto-display via `_repr_html_`

---

## 8. Weighted Least Squares (DONE)

- [x] `ols(..., weights=)` for analytic weights (aweight)
- [x] Frequency weights (fweight) support
- [x] WLS with FE absorption and clustered SEs

---

## 9. Bootstrap Standard Errors (DONE)

- [x] `ols(..., vcov="bootstrap", n_boot=999)`
- [x] Wild bootstrap (Webb 6-point) for clustered data
- [x] Pairs bootstrap
- [x] Works with all estimators: OLS, 2SLS, LIML, GMM, panel FE/RE/FD

---

## 10. Limited Dependent Variable Models (DONE)

### 10a. Probit
- [x] `probit("y ~ x1 + x2", data=df)` via MLE
- [x] Marginal effects (at means / average)
- [x] Robust and clustered SEs

### 10b. Logit
- [x] `logit("y ~ x1 + x2", data=df)` via MLE
- [x] Odds ratios option
- [x] Robust and clustered SEs

---

## 11. Dynamic Panel GMM (Arellano-Bond) (DONE)

- [x] `panel_ab("y ~ x1 + x2", data=df, entity=, time=, lags=)`
- [x] System GMM (Blundell-Bond)
- [x] Sargan/Hansen test for overidentification
- [x] AR(1)/AR(2) serial correlation tests

---

## 12. Quantile Regression (DONE)

- [x] `quantreg("y ~ x1 + x2", data=df, tau=0.5)` (median regression)
- [x] Multiple quantiles: tau=[0.25, 0.5, 0.75]
- [x] Bootstrap SEs for inference

---

## 13. Out-of-Sample Prediction (DONE)

- [x] `result.predict(new_df)` on `RegressionResult`
- [x] Handle intercept, interactions, indicator dummies (`col=level` format)
- [x] `result.predict_interval(new_df, alpha=0.05)` — fit, se, lower, upper
- [x] Indicator-continuous interaction support (`i.group:x1`)

---

## 14. fixest-Style IV Formula Syntax (DONE)

- [x] Explicit `||` pre-processing (normalizes to `| |` before split)
- [x] Support `y ~ x1 | fe1 | x_endog ~ z1 + z2` (single `|` separators)
- [x] Both `||` and `| |` produce identical FormulaSpec
- [x] Comprehensive formula parser tests (28 total)

---

## 15. Poisson Pseudo-Maximum Likelihood (PPML) (DONE)

- [x] `ppml("y ~ x1 + x2", data=df)` via Newton-Raphson/IRLS
- [x] Robust (HC1) and clustered SEs (sandwich VCV, pseudo-ML)
- [x] Separation detection (warns on |beta| > 10 or mu > 1e10)
- [x] Pseudo R² via deviance ratio

---

## 16. Coefficient Plots (DONE)

- [x] `result.coefplot()` — matplotlib coefficient plot with CIs
- [x] Customizable: variable selection, exclude, horizontal/vertical
- [x] `coefplot(*results, labels=)` — multi-model overlay

---

## 17. Partial Regression (Added-Variable) Plots (DONE)

- [x] `result.avplot(var)` — added-variable plot for a single regressor
- [x] Residualize y and x on all other regressors, scatter + fitted line
- [x] `result.avplot()` — grid of all regressors (skips `_cons`)
- [x] FWL theorem verified: slope matches full regression coefficient
