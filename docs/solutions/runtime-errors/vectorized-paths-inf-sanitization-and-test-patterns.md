---
title: "Vectorized NumPy paths must sanitize inf independently + CI test patterns"
category: runtime-errors
tags: [inf, nan, sanitization, vectorized, numpy, fama-macbeth, ci, pytest, optional-dependencies, conventions]
module: _fama_macbeth.py, tests/
symptom: >
  (1) Fama-MacBeth second pass produces corrupted coefficients when input
  data contains inf/-inf values — NaN-only masking misses inf.
  (2) CI fails with ModuleNotFoundError for altair in plot tests.
  (3) Shanken correction produces wrong results when intercept indexing
  assumes _cons is first instead of last.
root_cause: >
  (1) Vectorized NumPy paths bypass extract_arrays(), skipping inf→null
  conversion. (2) Tests for optional dependencies lack importorskip guard.
  (3) Project convention puts _cons LAST but pseudocode assumed first.
---

# Vectorized Paths, CI Test Patterns, and Intercept Convention

Three learnings from the rolling_reg + fama_macbeth implementation session (2026-03-16).

## Learning 1: Vectorized NumPy paths must sanitize inf independently

### Problem

`extract_arrays()` in `_utils.py` is the single data cleaning chokepoint — it converts NaN and inf to null, then drops nulls. Any code that builds NumPy arrays directly from Polars data (bypassing `extract_arrays()`) misses the inf sanitization.

The Fama-MacBeth second pass builds a return matrix `Y` by iterating Polars rows into a pre-allocated NumPy array. This path only checked `np.isnan()`, but `np.isnan(np.inf)` is `False` — so inf values survived into OLS solves, silently corrupting all lambda estimates.

### Fix

Before any vectorized linear algebra on directly-constructed NumPy arrays:

```python
# Y is a NumPy array built directly from Polars columns (NOT via extract_arrays)
Y[~np.isfinite(Y)] = np.nan
# np.isfinite catches both NaN and inf in one call
```

### General Rule

**Any time you see `df.select(...).to_numpy()` or manually stacking Polars columns into a NumPy matrix, you MUST add the `~np.isfinite` guard.** The only safe path that doesn't need this is going through `extract_arrays()`.

This applies to:
- Second-pass regressions (Fama-MacBeth cross-sectional OLS)
- Rolling windows building per-window arrays
- Any vectorized computation that constructs arrays outside the normal estimation pipeline

### Related

- `docs/solutions/runtime-errors/fe-singleton-contiguity-and-edge-case-guards.md` — Layer 1 covers NaN→null conversion at the Polars↔NumPy boundary
- `docs/solutions/runtime-errors/polars-reg-comprehensive-code-review.md` — Prevention strategy #1: "extract_arrays should accept a single canonical list... No downstream code should subset rows independently"

---

## Learning 2: pytest.importorskip() for optional dependency tests

### Problem

CI environments install only core dependencies. Tests calling `plot_coefs()` or `plot_lambdas()` (which depend on altair) fail with `ModuleNotFoundError`, breaking the entire test suite even though the core package works fine.

### Fix

Add `pytest.importorskip()` at the start of each test that uses an optional dependency:

```python
def test_plot_coefs():
    pytest.importorskip("altair")
    # ... test code that calls plot_coefs() ...

def test_something_with_pandas():
    pytest.importorskip("pandas")
    # ... test code ...
```

This marks the test as "skipped" (not "failed") when the dependency is missing.

### Existing Pattern

Already used in:
- `test_rolling.py:164` — altair
- `test_plotting.py:4` — altair (module-level)
- `test_grs.py:250` — pandas
- `test_compare.py:12-14` — pyfixest, statsmodels, linearmodels

**Every new test that touches an optional dependency must use this pattern.**

---

## Learning 3: Intercept convention — `_cons` is LAST

### Problem

polars_reg places `_cons` (the intercept) as the **last** element in `names` and `coefficients`. Code that assumes intercept-first indexing (e.g., `mean_lambda[1:]` for slopes) silently produces wrong results.

The Shanken correction pseudocode initially used `mean_lambda[1:]` — this incorrectly excluded the first factor and included the intercept in the slope vector.

### Fix

Always use `[:-1]` to exclude the intercept (get slopes) and `[-1]` to get the intercept:

```python
# WRONG — assumes intercept is first (index 0):
slopes = mean_lambda[1:]
Sigma_aug[1:, 1:] = Sigma_f

# CORRECT — intercept is last (index -1):
slopes = mean_lambda[:-1]
Sigma_aug[:-1, :-1] = Sigma_f
intercept = mean_lambda[-1]
```

When building augmented matrices (adding a constant column), the constant goes last:

```python
X_aug = np.column_stack([factor_returns, np.ones(T)])
# names = [...factor_names..., "_cons"]
```

### Prevention

Any new estimator, diagnostic, or post-estimation function that separates slopes from intercept must follow this convention. Grep for `[1:]` on coefficient arrays and verify it shouldn't be `[:-1]`.
