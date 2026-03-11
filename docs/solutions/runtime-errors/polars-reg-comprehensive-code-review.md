---
title: "Comprehensive code review: 82 issues fixed across polars_reg"
date: 2026-03-11
category: runtime-errors
tags:
  - code-review
  - edge-cases
  - fixed-effects
  - demeaning
  - standard-errors
  - degrees-of-freedom
  - logit
  - LIML
  - GMM
  - bootstrap
  - test-quality
severity: critical
component:
  - _demean.py
  - _ols.py
  - _gmm.py
  - _iv.py
  - _panel.py
  - _se.py
  - _formula.py
  - _utils.py
  - _binary.py
  - _quantile.py
  - _ppml.py
  - _diagnostics.py
  - _results.py
  - _regtable.py
  - _arellano_bond.py
  - __init__.py
  - stata.py
  - r_equiv.py
symptoms:
  - "RuntimeError on two-way FE when singleton removal empties all observations"
  - "Silent wrong degrees of freedom for models with 3+ fixed effects"
  - "Logit marginal effects sign error"
  - "LIML bootstrap incorrectly using OLS instead of LIML"
  - "GMM multi-way clustered VCV computed incorrectly"
  - "Crashes on duplicate column names in formula"
  - "Crashes on null values in fixed-effect columns"
  - "Non-deterministic test failures due to unseeded RNG"
status: resolved
---

# Comprehensive Code Review: 82 Issues Fixed

**Trigger:** Runtime crash using two-way fixed effects, tracing to `_demean.py`.

**Scope:** Full review of all 52 Python files (20 source, 32 test). 82 issues identified and fixed across 31 files (375 insertions, 165 deletions). All 437 tests pass.

---

## Root Cause

`codes.max()` in `_demean.py` at line 98 crashes on an empty NumPy array when `drop_singletons` removes all observations in a two-way FE model. No empty-array guard existed. The broader investigation revealed 81 additional issues, 16 of which were critical.

## Investigation

1. Deployed 12 parallel code reviewer subagents across all 52 Python files
2. Each reviewer logged findings to `.claude/logs/2026-03-11_code-review/`
3. Consolidated into a single report organized by severity (summary.md)
4. Deployed 10 parallel fix agents on non-overlapping file sets

## Solution

### Critical source fixes (16)

1. **`_demean.py`** -- Empty-array guard before `codes.max()` (the original crash). Re-contiguified codes after singleton removal.
2. **`_demean.py`** -- Corrected `absorbed_dof` for 3+ FE: replaced pairwise-max heuristic with multipartite graph via `scipy.sparse.csgraph.connected_components`.
3. **`_utils.py`** -- Duplicate column detection in `drop_nulls`. FE/cluster columns included in null-drop set.
4. **`_ols.py`** -- Rust FE path `fitted()`/`predict()` no longer returns zeros. Frequency weights filtered during singleton drop.
5. **`_gmm.py`** -- GMM multi-way cluster raises `NotImplementedError` instead of silently using wrong VCV. LIML bootstrap uses LIML re-estimation (was OLS).
6. **`_panel.py`** -- `panel_fd` with `vcov="iid"` no longer silently switches to clustered. `panel_re` entity column included in null-drop.
7. **`_binary.py`** -- Logit marginal effects Jacobian sign corrected.
8. **`_diagnostics.py`** -- `_matrix_power` handles singular matrices.
9. **`__init__.py`** -- Silent `ImportError` no longer swallowed.
10. **`stata.py`** -- Multi-way cluster raises instead of truncating.

### Test fixes

- Deterministic RNG seeds, removed 9 unnecessary `@pl.StringCache()` decorators
- New tests: reghdfe DFC formula, `ensure_polars`, formula error paths
- Strengthened weak assertions, removed dead `skipif` guards
- Fixed flaky PPML warning tests, tautological predict test

## Verification

437 tests pass, 17 skipped, 0 failures. Original two-way FE crash resolved. All Stata parity tests continue to pass.

---

## Lessons Learned

### Edge cases are the norm in econometrics

The largest bug cluster (C1-C3, C16) stems from code that works on typical datasets but crashes on structural peculiarities -- singleton FE, duplicate columns from interactions, non-contiguous codes after filtering. In econometric workflows with firm-year FE, these arise routinely.

### Silent wrong answers are worse than crashes

Nearly half the critical bugs (C4-C12, C15) produce silently incorrect results. Wrong DoF formulas, misaligned weight vectors, bootstrap procedures calling the wrong estimator -- each could cause a published paper to report wrong standard errors. A crash is a gift; silent corruption is the real enemy.

### Tests that pass are not tests that protect

The test suite had quantity but not quality where it mattered. Tautological assertions testing algebraic identities, non-deterministic RNG, and missing coverage for the exact formulas that differentiate this package (reghdfe DFC corrections) meant the hardest code was the least tested.

### Rust acceleration introduced a correctness boundary

The Rust native extension returning zeros for fitted values (C6) illustrates the risk: fast path and slow path must produce identical results, and both paths must be tested in CI.

---

## Prevention Strategies

### 1. Null-safety by construction

`extract_arrays` should accept a single canonical list of all participating columns and perform null-dropping once. No downstream code should subset rows independently. After extraction, row count is fixed.

### 2. Fail loudly on unsupported combinations

Every function should validate arguments early and raise `ValueError`/`NotImplementedError` for unsupported combinations rather than falling through to wrong logic.

### 3. Deterministic test seeding

Replace all `np.random.randn(...)` with explicit `np.random.default_rng(seed)`. For bootstrap tests, either fix the seed or test weak properties that hold with overwhelming probability.

### 4. Oracle-based assertions

Every test assertion should compare against an independent source (Stata, R, analytical formula), never against the code's own intermediate results.

### 5. Dual-path Rust/Python testing

Parameterize tests over `backend=["python", "rust"]` to ensure both paths are exercised in CI.

### 6. Defensive assertions at module boundaries

After `extract_arrays`: all arrays same row count, FE codes contiguous, no NaN. After `demean`: same shape, no NaN/Inf. After VCV: symmetric, PSD. Gate behind `debug=True`.

### 7. Formula coverage tracking

Maintain a document listing every DoF correction and VCV scaling formula with a pointer to its validating test. Missing pointers are known risks.

---

## Related Documentation

### Review logs
- `.claude/logs/2026-03-11_code-review/summary.md` -- Master 82-issue report
- `.claude/logs/2026-03-11_code-review/review_*.md` -- 12 per-module reviews

### SE investigations (preceded this review)
- `.claude/logs/2026-03-10_se-investigations/` -- 5 SE discrepancy analyses (IV, panel FD, panel NW, probit, quantile)

### Cross-references
- **C10 (panel_fd vcov routing) relates to `panel_fd_se_review.md`**: the Mar 10 investigation called it a "design choice"; the code review found it was actually a fall-through bug.
- **C5 (absorbed_dof) contradicts CLAUDE.md**: CLAUDE.md documented "pairwise connected components" which was the wrong algorithm for 3+ FE. Now fixed to use multipartite graph.
- **C8 (GMM multi-way cluster) vs progress-log.md**: progress log claims "CGM multi-way clustering supports arbitrary N dimensions" -- true in `_se.py` but was never implemented in `_gmm.py`.
