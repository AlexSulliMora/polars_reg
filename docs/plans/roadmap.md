# polars_reg Roadmap

## 1. R Equivalence Layer

Mirror the existing Stata equivalence system for R users.

### 1a. `to_r()` — Generate equivalent R code
- [ ] Add `polars_reg/r_equiv.py` with `to_r()` function (same signature as `to_stata()`)
- [ ] Translation mapping:
  - `ols` → `lm(y ~ x1 + x2, data=df)` (base R)
  - `ols` + robust → `coeftest(model, vcov=vcovHC(model, type="HC1"))` (sandwich/lmtest)
  - `ols` + cluster → `coeftest(model, vcov=vcovCL(model, cluster=df$firm_id))` (sandwich)
  - `ols` + FE → `feols(y ~ x1 + x2 | fe1 + fe2, data=df, vcov=~firm_id)` (fixest)
  - `ols` + multi-way cluster → `feols(..., vcov=~firm_id + year)` (fixest)
  - `iv2sls` → `ivreg(y ~ x1 + x2 | x_endog | z1 + z2, data=df)` (AER) or `feols(y ~ x1 + x2 | x_endog ~ z1 + z2, data=df)` (fixest)
  - `iv2sls` + FE → `feols(y ~ x1 + x2 | fe1 | x_endog ~ z1 + z2, data=df)` (fixest)
  - `liml` → `ivreg(y ~ x1 + x_endog | z1 + z2, data=df, model="liml")` (AER)
  - `gmm_iv` → note: no direct single-line R equivalent; document `gmm` package usage
  - `panel_fe` → `plm(y ~ x1 + x2, data=pdf, model="within", index=c("firm","year"))` (plm)
  - `panel_re` → `plm(..., model="random")` (plm)
  - `panel_fd` → `plm(..., model="fd")` (plm)
- [ ] Include library() calls in output (e.g., `library(fixest)`)
- [ ] Add `pyr=True` option that wraps in `reticulate` or `rpy2` Python code for automated comparison
- [ ] Export `to_r` from `__init__.py`

### 1b. `compare_r()` — Run both and compare (via rpy2)
- [ ] Add `compare_r()` to `polars_reg/r_equiv.py`
- [ ] Use rpy2 to execute R code and extract coefficients/SEs when available
- [ ] Fall back gracefully when rpy2 is not installed (same pattern as pystata fallback)
- [ ] Return `ComparisonReport` (reuse from `stata.py` or shared base)

### 1c. Update showcase notebook
- [ ] Add Section 10: R Equivalence showing `to_r()` output for all estimator types
- [ ] Add `compare_r()` example (with graceful fallback)

### 1d. Tests for R translation
- [ ] `tests/test_r_equiv.py` — unit tests for `to_r()` translation (no R needed)
- [ ] Verify each estimator produces syntactically correct R code
- [ ] Test edge cases: no intercept, multi-way cluster, FE + IV

---

## 2. Cross-Validated Parity Tests (Stata + R)

Extend the existing `tests/stata_compare.py` infrastructure to also verify against R.

### 2a. R parity infrastructure
- [ ] Add `tests/r_compare.py` mirroring `tests/stata_compare.py`
  - `r_available()` — check if Rscript is on PATH and fixest/sandwich are installed
  - `to_r_command()` — translate polars_reg calls to R scripts
  - `_run_r_script()` — execute R script, parse CSV output of coefficients/SEs
  - `assert_r_parity()` — all-in-one comparison function
- [ ] R scripts should: read CSV data, run regression, write results CSV
- [ ] Handle R package differences:
  - fixest for FE/clustering (primary target — closest to reghdfe)
  - AER::ivreg for basic IV
  - plm for panel estimators

### 2b. Combined parity test file
- [ ] Add `tests/test_cross_parity.py` that tests against both Stata and R
- [ ] Test matrix (each row = one test, check against available backends):

  | Estimator | Specification | Stata match | R match | Notes |
  |-----------|--------------|-------------|---------|-------|
  | OLS | iid SE | ✓ | ✓ | |
  | OLS | HC1 | ✓ | ✓ | R: sandwich::vcovHC type="HC1" |
  | OLS | HC2 | ✓ | ✓ | |
  | OLS | HC3 | ✓ | ✓ | |
  | OLS | cluster(firm) | ✓ | ✓ | R: fixest::feols |
  | OLS | no intercept | ✓ | ✓ | |
  | reghdfe | 1-way FE + cluster | ✓ | ✓ | R: fixest::feols |
  | reghdfe | 2-way FE + cluster | ✓ | ✓ | |
  | reghdfe | 2-way FE + 2-way cluster | ✓ | ✓ | |
  | reghdfe | 2-way FE + iid | ✓ | ✓ | R: fixest vcov="iid" |
  | 2SLS | iid | ✓ | ✓ | R: AER::ivreg |
  | 2SLS | HC1 | ✓ | ✓ | |
  | LIML | iid | ✓ (loose) | ✓ (loose) | Eigenvalue solver differences |
  | GMM | robust | ✓ (medium) | — | No direct R single-fn equivalent |
  | panel_fe | cluster(entity) | — | ✓ | R: plm model="within" |
  | panel_re | iid | — | ✓ | R: plm model="random" |
  | panel_fd | cluster(entity) | — | ✓ | R: plm model="fd" |

- [ ] Known Stata vs R differences to document:
  - **DoF corrections**: fixest and reghdfe may differ slightly in cluster dfc for nested FE. Match Stata (reghdfe) when they disagree.
  - **LIML eigenvalue**: different solvers → loosen tolerance to 2e-3
  - **GMM**: R's `gmm` package uses different defaults than `ivregress gmm`. Skip R comparison or use wide tolerance.
  - **Panel RE**: plm and xtreg use same Swamy-Arora but may differ in small-sample corrections
  - **R² definition**: fixest reports within-R² for FE models (same as reghdfe). plm may report different R² variants.

### 2c. Tolerance hierarchy
- [ ] Priority: match Stata first, then R. When Stata and R disagree, document why and match Stata.
- [ ] Tolerance tiers (carry forward from existing):
  - TIGHT (1e-6): OLS, 2SLS — closed-form solutions
  - REGHDFE (1e-5): FE absorption — iterative algorithms differ
  - MEDIUM (1e-4): GMM, panel — implementation differences in weighting/corrections
  - LOOSE (2e-3): LIML — eigenvalue solver sensitivity

---

## 3. GroupBy Regression

Accept a Polars `GroupBy` object to run the same regression per group and collect results — e.g., estimating factor loadings for each stock, or running regressions per industry.

### 3a. Core API
- [ ] `pr.ols("y ~ x1 + x2", data=df.group_by("ticker"))` returns a collection of results keyed by group
- [ ] Should also work with `iv2sls`, `liml`, `gmm_iv`, `panel_fe`, etc.
- [ ] Return type: `GroupRegressionResult` (dict-like, keyed by group values)
  - `.keys()` — group labels
  - `[group]` — individual `RegressionResult`
  - `.coef_table()` — stacked Polars DataFrame with a group column
  - `.summary()` — compact multi-group summary
- [ ] Parallel execution via Polars' thread pool or Python multiprocessing

### 3b. Integration with regtable
- [ ] `regtable(*group_result.values())` should work naturally
- [ ] Auto-label columns with group names

### 3c. Edge cases
- [ ] Groups with too few observations → skip with warning, don't crash
- [ ] Groups with singular X'X → skip with warning
- [ ] Consistent column ordering across groups (some groups may lack certain categories)

---

## 4. Pandas Compatibility

Accept `pandas.DataFrame` as input — convert to Polars internally, run as normal, and return Polars-native results. Pandas users can still use `.coef_table().to_pandas()` etc.

### 4a. Input handling
- [ ] In each estimator (`ols`, `iv2sls`, `liml`, `gmm_iv`, `panel_fe`, `panel_re`, `panel_fd`), check `isinstance(data, pd.DataFrame)` at the top and convert via `pl.from_pandas(data)`
- [ ] Centralize in a shared helper (e.g., `_utils.py: ensure_polars(data)`) to avoid repeating the check in every function
- [ ] Import pandas only inside the check (`if isinstance(...)`) so pandas remains an optional dependency
- [ ] No changes to return types — `RegressionResult` stays the same (NumPy arrays, Polars `.coef_table()`)

### 4b. Tests
- [ ] Test that `ols("y ~ x1", data=pandas_df)` produces identical results to the Polars version
- [ ] Test that pandas is truly optional — importing polars_reg without pandas installed doesn't error

---

## 5. Other Feature Gaps

### 5a. Interaction terms in formula
- [ ] Support `x1:x2` syntax for interaction terms in formula parser
- [ ] Support `x1##x2` (full factorial: main effects + interaction)

### 5b. HAC / Driscoll-Kraay standard errors
- [ ] Newey-West (HAC) for time series
- [ ] Driscoll-Kraay for panel data with cross-sectional dependence

### 5c. Additional diagnostics
- [ ] Wald test for linear restrictions
- [ ] Hausman test (FE vs RE)
- [ ] Weak instrument diagnostics (Stock-Yogo critical values, Kleibergen-Paap)

---

## 6. Polish & Packaging

### 6a. Documentation
- [ ] README with quickstart, installation, feature overview
- [ ] API reference (auto-generated from docstrings)
- [ ] Add showcase notebook link to README

### 6b. CI/CD
- [ ] GitHub Actions: run tests on push (Python 3.10+)
- [ ] Optional Stata parity job (manual trigger, requires Stata license)
- [ ] Optional R parity job (install fixest, AER, plm, sandwich, lmtest)
- [ ] Auto-publish to PyPI on tagged release

### 6c. Performance
- [ ] Benchmark suite (N = 1K, 10K, 100K, 1M)
- [ ] Sparse FE dummies for large group counts (>1000)
- [ ] Profile demeaning for bottlenecks
