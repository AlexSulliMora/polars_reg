---
title: "Test Edge Case and Robustness Coverage"
type: feat
date: 2026-03-11
---

# Test Edge Case and Robustness Coverage

## Overview

The test suite has 440 tests across 28 files but critical edge cases slip through because tests exercise happy paths with clean, balanced data. The singleton FE bug (non-contiguous codes after `drop_singletons`) survived because `test_drop_singletons` tested the mask but never demeaned the result, and every demeaning test used perfectly contiguous codes. This plan adds systematic edge-case coverage at every level: sub-functions, internal pipelines, and public API.

## Problem Statement

From the code review (82 issues, 16 critical): "Tests that pass are not tests that protect." Specific gaps:

- **No null/NaN/Inf tests** for X, FE, cluster, weight, or instrument columns
- **No degenerate input tests** — empty DataFrame, 1 observation, all singletons, all nulls
- **No collinearity tests** — constant columns, perfect multicollinearity, near-singular X'X
- **No pipeline integration tests** — sub-functions tested in isolation but seams between them untested
- **No type diversity** — all test data is float64; no int, bool, categorical, or LazyFrame inputs
- **No dual-path Rust/Python** — Rust native extension has different numerical behavior but is never compared
- **9 estimators lack predict() tests**, LIML/GMM lack FE tests, panel_ab/sys_gmm lack pandas compat

## Conventions (from repo analysis)

- File per module: `test_{module}.py`
- RNG: always `np.random.default_rng(seed)`, never bare `np.random`
- Assertions: `np.testing.assert_allclose(actual, expected, atol=X)` or plain `assert`
- Error paths: `pytest.raises(ValueError, match="substring")`
- No `@pytest.mark.parametrize`, no mocking — explicit test functions, real computations
- Section headers: `# -- Section Name ----------------------------------------`
- Docstrings: required, brief one-liner
- Fixture sizes: N=200-1000 basic, N=50-200 panel

---

## Phase 1: Input Validation and Degenerate Data

**Goal:** Every public function handles garbage in gracefully — either processes correctly or raises a clear error.

### 1.1 `tests/test_utils.py` — Input extraction edge cases

| Test | What it covers |
|------|---------------|
| `test_extract_arrays_nan_in_y` | NaN values in dependent variable are dropped |
| `test_extract_arrays_nan_in_x` | NaN in regressor columns are dropped |
| `test_extract_arrays_nan_in_fe` | NaN in FE column raises or drops correctly |
| `test_extract_arrays_nan_in_cluster` | NaN in cluster column handled |
| `test_extract_arrays_inf_in_x` | Inf/-Inf in numeric columns raises ValueError |
| `test_extract_arrays_empty_df` | 0-row DataFrame raises ValueError |
| `test_extract_arrays_single_row` | 1-row DataFrame works or raises meaningful error |
| `test_extract_arrays_all_null_column` | Column with all nulls raises ValueError |
| `test_extract_arrays_int_columns` | Integer-typed x columns auto-cast to float64 |
| `test_extract_arrays_bool_columns` | Boolean columns cast correctly |
| `test_extract_arrays_missing_column` | Formula references non-existent column raises KeyError/ValueError |
| `test_extract_arrays_duplicate_columns` | Duplicate column names in formula raises ValueError |
| `test_ensure_polars_passthrough` | pl.DataFrame passes through unchanged |
| `test_ensure_polars_lazyframe` | pl.LazyFrame gets collected |
| `test_ensure_polars_pandas` | pandas DataFrame converts correctly |
| `test_ensure_polars_invalid` | Non-DataFrame input raises TypeError |

### 1.2 `tests/test_demean.py` — Degenerate FE cases

| Test | What it covers |
|------|---------------|
| `test_demean_empty_array` | 0-row input returns empty array |
| `test_demean_single_observation` | 1 observation with FE |
| `test_demean_all_singletons_removed` | All obs are singletons → empty after drop → handled gracefully |
| `test_demean_constant_column` | Column with zero variance after demeaning |
| `test_demean_extreme_values` | Values near 1e15 don't overflow |
| `test_demean_many_fe_levels` | 500+ FE levels with N=1000 (near-saturation) |
| `test_demean_weighted_singletons` | Weighted demeaning with singletons dropped + reindexed |
| `test_absorbed_dof_three_way` | 3-way FE absorbed DoF matches brute-force LSDV |
| `test_absorbed_dof_empty` | Empty fe_dict returns 0 |
| `test_drop_singletons_cascading` | Removing one singleton creates new singletons → cascade |
| `test_drop_singletons_all_removed` | Every observation is a singleton → empty mask |
| `test_reindex_fe_codes_already_contiguous` | Contiguous codes pass through unchanged |
| `test_reindex_fe_codes_empty` | Empty dict returns empty dict |

### 1.3 `tests/test_formula.py` — Formula error paths

| Test | What it covers |
|------|---------------|
| `test_formula_empty_string` | Empty string raises ValueError |
| `test_formula_no_rhs` | `"y ~"` raises or returns empty exog |
| `test_formula_duplicate_vars` | `"y ~ x1 + x1"` deduplicates |
| `test_formula_same_var_exog_and_fe` | `"y ~ x1 | x1"` raises or warns |
| `test_formula_same_var_endog_and_exog` | `"y ~ x1 || x1 ~ z1"` raises ValueError |
| `test_formula_special_chars_in_names` | Column names with dots, underscores, numbers |

---

## Phase 2: Sub-Function Unit Tests

**Goal:** Test internal functions at module boundaries — the seams where bugs hide.

### 2.1 `tests/test_se.py` — VCV edge cases

| Test | What it covers |
|------|---------------|
| `test_vcov_iid_near_singular` | Near-singular X'X produces warning or raises |
| `test_vcov_robust_single_cluster` | 1 cluster group raises or warns (G=1 makes dfc = 0/(0-1)) |
| `test_vcov_clustered_singleton_cluster` | Cluster where one group has 1 obs |
| `test_vcov_multiway_two_identical_dims` | Two cluster dimensions that are identical |
| `test_vcov_hac_single_time_period` | T=1 for HAC raises or returns iid equivalent |
| `test_vcov_dk_single_entity` | N=1 entity for Driscoll-Kraay |
| `test_vcov_all_variants_finite` | Every VCV function returns finite symmetric PSD matrix |

### 2.2 `tests/test_diagnostics.py` — Diagnostic edge cases

| Test | What it covers |
|------|---------------|
| `test_wald_test_singular_R` | Singular restriction matrix R |
| `test_hausman_test_identical_models` | Two identical models (diff = 0) |
| `test_weak_instrument_single_instrument` | Exactly-identified case (1 endog, 1 instrument) |
| `test_kleibergen_paap_singular_matrix` | Near-singular concentration matrix |
| `test_matrix_power_singular` | `_matrix_power(A, -0.5)` with zero eigenvalue |

### 2.3 `tests/test_panel.py` — Panel sub-function coverage

| Test | What it covers |
|------|---------------|
| `test_panel_fe_single_entity` | Panel with 1 entity (all variation is within) |
| `test_panel_re_theta_computation` | Swamy-Arora theta is between 0 and 1 |
| `test_panel_fd_single_time_period` | First-difference with T=2 (minimum) |
| `test_panel_fe_constant_regressor` | Regressor that is constant within entity |
| `test_panel_unbalanced_extreme` | Some entities have 2 obs, others have 50 |

---

## Phase 3: Public API Robustness

**Goal:** Every user-facing function handles messy real-world data without crashing or silently producing wrong results.

### 3.1 Shared robustness fixture

Add to `tests/conftest.py`:

```python
@pytest.fixture
def messy_data():
    """DataFrame with nulls, extreme values, singletons, and mixed types."""
    rng = np.random.default_rng(777)
    n = 500
    df = pl.DataFrame({
        "y": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "x_const": np.ones(n),              # zero variance
        "x_with_nan": np.where(rng.random(n) < 0.05, np.nan, rng.standard_normal(n)),
        "fe1": rng.integers(0, 50, size=n),
        "fe2": rng.integers(0, 30, size=n),
        "cluster1": rng.integers(0, 20, size=n),
        "entity": np.repeat(np.arange(50), 10),
        "time": np.tile(np.arange(10), 50),
        "z1": rng.standard_normal(n),       # instrument
        "z2": rng.standard_normal(n),       # instrument
        "x_endog": rng.standard_normal(n),  # endogenous
        "binary_y": rng.integers(0, 2, size=n),
        "count_y": rng.poisson(3, size=n).astype(float),
    })
    # Force singletons in FE
    df = df.with_columns([
        pl.when(pl.arange(0, n) < 3).then(pl.lit(997 + pl.arange(0, n))).otherwise(pl.col("fe1")).alias("fe1"),
    ])
    return df
```

### 3.2 `tests/test_ols.py` — OLS robustness

| Test | What it covers |
|------|---------------|
| `test_ols_nan_in_x_dropped` | NaN rows in x are dropped, results are finite |
| `test_ols_constant_regressor_raises` | `"y ~ x_const"` with only a constant column |
| `test_ols_intercept_only_with_fe` | `"y ~ 1 | fe1"` — no regressors, only FE |
| `test_ols_lazyframe_input` | `pl.LazyFrame` input works identically to DataFrame |
| `test_ols_integer_columns` | Integer-typed x and y columns work |
| `test_ols_all_vcov_variants_finite` | Loop over iid/HC0/HC1/HC2/HC3/cluster — all produce finite SEs |
| `test_ols_predict_after_fe` | `predict()` and `fitted()` return correct values with FE |
| `test_ols_large_fe_ratio` | 200 FE levels with N=300 (near-saturated model) |

### 3.3 `tests/test_iv.py` — IV robustness

| Test | What it covers |
|------|---------------|
| `test_iv2sls_nan_dropped` | NaN in instruments handled |
| `test_iv2sls_weak_instrument` | Near-zero first-stage F — results still finite |
| `test_iv2sls_exact_identification` | k_endog = k_instruments (just-identified) |
| `test_iv2sls_lazyframe` | LazyFrame input |
| `test_liml_with_fe` | LIML + absorbed FE |
| `test_gmm_with_fe` | GMM-IV + absorbed FE (if supported, else verify raises) |

### 3.4 `tests/test_panel.py` — Panel robustness

| Test | What it covers |
|------|---------------|
| `test_panel_fe_nan_in_entity` | NaN in entity column |
| `test_panel_re_lazyframe` | LazyFrame input |
| `test_panel_fd_two_periods_only` | Minimum viable first-difference |
| `test_panel_ab_lazyframe` | LazyFrame input for Arellano-Bond |
| `test_panel_fe_iid_se_explicit` | `cluster=[]` forces iid SEs |

### 3.5 `tests/test_binary.py` — Binary model robustness

| Test | What it covers |
|------|---------------|
| `test_probit_perfect_separation` | All y=1 for x>0 — perfect separation detection |
| `test_logit_nan_dropped` | NaN in x columns handled |
| `test_probit_lazyframe` | LazyFrame input |
| `test_marginal_effects_finite` | Marginal effects are all finite |
| `test_odds_ratios_positive` | Odds ratios are all > 0 |

### 3.6 Other estimators

| Test file | Test | What it covers |
|-----------|------|---------------|
| `test_quantile.py` | `test_quantreg_nan_dropped` | NaN handling |
| `test_quantile.py` | `test_quantreg_lazyframe` | LazyFrame input |
| `test_ppml.py` | `test_ppml_nan_dropped` | NaN handling |
| `test_ppml.py` | `test_ppml_lazyframe` | LazyFrame input |
| `test_arellano_bond.py` | `test_panel_ab_pandas_compat` | pandas DataFrame input |
| `test_arellano_bond.py` | `test_panel_sys_gmm_pandas_compat` | pandas DataFrame input |
| `test_groupby.py` | `test_groupby_nan_in_group_col` | NaN in group-by column |
| `test_groupby.py` | `test_groupby_empty_group` | Group with 0 valid obs |
| `test_regtable.py` | `test_regtable_empty_results` | Empty result list |
| `test_regtable.py` | `test_regtable_mismatched_models` | Models with different variable sets |

---

## Phase 3.7: Code Guards Required by Tests

Several tests in this plan will FAIL against the current code because the code lacks guards. These guards must be added alongside the tests:

| Guard | Location | Why |
|-------|----------|-----|
| **NaN-to-null conversion** | `_utils.py` `extract_arrays()`, before `drop_nulls()` | IEEE NaN passes through `drop_nulls()` and silently poisons all downstream linear algebra. Add `pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c))` for float cols. |
| **G=1 cluster guard** | `_se.py` `vcov_clustered()` | `dfc = G/(G-1)` = `1/0` = inf when single cluster group. Raise `ValueError("Clustered SEs require at least 2 cluster groups")`. |
| **T=1 DK guard** | `_se.py` `vcov_driscoll_kraay()` | `T/(T-1)` = `1/0` when single time period. Raise `ValueError`. |
| **All-singletons guard** | `_ols.py`, `_iv.py`, `_panel.py` after `drop_singletons()` | `keep.sum() == 0` → raise `ValueError("All observations dropped as singletons")`. |
| **Collinearity wrapper** | `_ols.py`, `_iv.py` around `np.linalg.solve` | Catch `LinAlgError` and re-raise with "Design matrix is singular. Check for collinearity." |
| **Empty DataFrame guard** | `_utils.py` `extract_arrays()` | 0-row input → raise `ValueError("DataFrame has no observations")` before any processing. |
| **LIML FE error test** | `_gmm.py` `liml()` | Verify `NotImplementedError` is raised for FE (already raised, just needs test). |
| **GMM multi-way cluster error test** | `_gmm.py` `gmm_iv()` | Verify `NotImplementedError` is raised (already raised, just needs test). |
| **Panel time gaps warning** | `_panel.py` `panel_fd()` | `shift(1).over(entity)` computes diffs between non-adjacent periods silently. Add warning or doc. |

---

## Phase 4: Cross-Path and Regression Tests

**Goal:** Rust and Python paths produce identical results; known bugs stay fixed.

### 4.1 `tests/test_dual_path.py` (new file)

| Test | What it covers |
|------|---------------|
| `test_ols_fe_python_vs_rust` | Force Python path (`_HAS_NATIVE=False`), compare to Rust path |
| `test_demean_python_vs_rust` | Compare demeaned arrays between paths |
| `test_ols_fe_predict_python_vs_rust` | `predict()` and `fitted()` match between paths |
| `test_clustered_se_python_vs_rust` | Clustered SE computation matches |

Implementation: monkeypatch `polars_reg._demean._HAS_NATIVE = False` for the Python path, then compare against default (Rust) path.

### 4.2 Regression tests (add to relevant files)

| Test | Prevents regression of |
|------|----------------------|
| `test_singleton_drop_demean_pipeline` | Non-contiguous FE codes after singleton removal (the original bug) |
| `test_all_singletons_error` | Empty array after total singleton removal (C1) |
| `test_fe_null_in_fe_column` | Null in FE column corruption (C4) |
| `test_gmm_multiway_cluster_raises` | GMM multi-way cluster silent wrong answer (C8) |
| `test_panel_fd_iid_vcov` | panel_fd vcov="iid" routing bug (C10) |

---

## Acceptance Criteria

- [ ] Every public API function has at least one test with NaN-containing input
- [ ] Every public API function has at least one test with LazyFrame input
- [ ] `drop_singletons → reindex → demean` pipeline has integration test
- [ ] All-singletons-removed case has test
- [ ] Empty DataFrame raises clear ValueError for every estimator
- [ ] 3+ FE absorbed DoF has test comparing against brute-force LSDV
- [ ] Every VCV variant (iid, HC0-3, cluster, NW, DK, bootstrap, wildboot) produces finite results
- [ ] Rust vs Python dual-path comparison test exists for OLS + FE
- [ ] No RuntimeWarning emitted during any test (enforced via `warnings.simplefilter("error", RuntimeWarning)`)
- [ ] All regression tests for the 16 critical bugs from the code review exist
- [ ] 0 test failures, 0 new warnings

---

## Phase 5: Additional Edge Cases (from SpecFlow analysis)

Tests surfaced by systematic flow analysis that aren't covered above:

| Test file | Test | What it covers |
|-----------|------|---------------|
| `test_results.py` | `test_confint_invalid_alpha` | `confint(alpha=0)` and `confint(alpha=1.5)` |
| `test_results.py` | `test_predict_interval_singular_vcv` | `predict_interval()` with near-singular VCV |
| `test_results.py` | `test_summary_long_variable_names` | Variable names > 14 chars don't break formatting |
| `test_diagnostics.py` | `test_hausman_no_common_coefficients` | Two models with disjoint variable sets |
| `test_diagnostics.py` | `test_kp_after_singleton_drop` | `kleibergen_paap_from_result()` when singletons were dropped |
| `test_binary.py` | `test_probit_perfect_separation_warning` | MLE divergence with perfect separation |
| `test_quantile.py` | `test_quantreg_extreme_tau` | `tau=0.01` and `tau=0.99` stability |
| `test_panel.py` | `test_panel_fd_time_gaps` | Non-consecutive time periods warn or handle correctly |
| `test_panel.py` | `test_panel_ab_minimum_periods` | T=2 (minimum for Arellano-Bond after differencing) |
| `test_groupby.py` | `test_groupby_null_in_group_column` | Null-keyed group from `group_by` |
| `test_groupby.py` | `test_groupby_group_fewer_obs_than_params` | Group with N < k |
| `test_se.py` | `test_vcov_hc2_high_leverage` | Hat diagonal near 1 with near-collinear data |
| `test_se.py` | `test_vcov_wildboot_single_cluster` | All obs in one cluster → zero bootstrap variance |
| `test_se.py` | `test_vcov_bootstrap_small_n` | N=5 with k=3 → most resamples singular |
| `test_utils.py` | `test_indicator_level_disappears_after_null_drop` | Indicator variable loses a level after null-dropping |
| `test_stata_equiv.py` | `test_to_stata_unsupported_estimator` | `to_stata()` with probit/quantreg raises |

## Design Decisions (Resolved)

1. **NaN handling policy**: IEEE NaN in float columns treated as null (row dropped, matching Stata `mvdecode`). Implement NaN-to-null conversion in `extract_arrays()` before `drop_nulls()`.
2. **G=1 cluster**: Raise `ValueError("Clustered SEs require at least 2 cluster groups")`.
3. **Python fallback testing**: Use `monkeypatch` on `_HAS_NATIVE = False` in a dedicated `test_dual_path.py` file.
4. **Minimum N per estimator**: Enforce minimum observation counts with clear error messages (e.g., "OLS requires at least k+1 observations").

## Test Count Estimate

~105-115 new tests across 13 files (11 existing + 1 new `test_dual_path.py` + conftest fixture).

## References

- Code review: `docs/solutions/runtime-errors/polars-reg-comprehensive-code-review.md`
- Review logs: `.claude/logs/2026-03-11_code-review/summary.md`
- Existing fixtures: `tests/conftest.py:1-91`
- Public API: `polars_reg/__init__.py`
