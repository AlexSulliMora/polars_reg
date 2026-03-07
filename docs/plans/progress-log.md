# polars_reg Implementation Progress Log

## Status: Complete (all 13 tasks done)

**67 tests passing** as of 2026-03-06

## Completed Tasks

### Phase 1: Scaffolding
- [x] Task 1.1: pyproject.toml, package skeleton, conftest fixtures (commit 3163995)

### Phase 2: Formula Parser
- [x] Task 2.1+2.2: FormulaSpec + parse_formula (OLS, FE, IV syntax) (commit 9212cd9)

### Phase 3: Utilities
- [x] Task 3.1: ExtractedArrays + extract_arrays (Polars-to-NumPy) (commit 3e09378)

### Phase 4: Standard Errors
- [x] Task 4.1-4.3: vcov_iid, vcov_robust (HC0-HC3), vcov_clustered, vcov_multiway_clustered (CGM) (commit d0bc365)

### Phase 5: Results + OLS
- [x] Task 5.1: RegressionResult dataclass with summary() (commit 8bcebf6)
- [x] Task 5.2: OLS estimator with iid/robust/clustered SEs (commit 4484fd7)

### Phase 6: Demeaning
- [x] Task 6.1-6.3: demean() with symmetric Kaczmarz + CG, drop_singletons, absorbed_dof (commit 4d932b2)

### Phase 7: reghdfe
- [x] Task 7.1: OLS with absorbed multi-way FE (commit aeb7361)

### Phase 8: IV/2SLS
- [x] Task 8.1-8.2: iv2sls() with first-stage F-stat (commit in _iv.py)

### Phase 9: LIML
- [x] Task 9.1: liml() in _gmm.py (commit 020804d)

### Phase 10: GMM
- [x] Task 10.1: gmm_iv() with Hansen J test (commit 020804d)

### Phase 11: Panel
- [x] Task 11.1: panel_fe() within estimator (commit 15e182c)

### Phase 12: Public API
- [x] Task 12.1-12.2: __init__.py exports + integration tests (commit 99f6d3d)

## Module Summary

| Module | Functions | Tests |
|--------|----------|-------|
| _formula.py | parse_formula | 6 |
| _utils.py | extract_arrays | 5 |
| _se.py | vcov_iid, vcov_robust, vcov_clustered, vcov_multiway_clustered | 9 |
| _results.py | RegressionResult | 6 |
| _ols.py | ols (with FE absorption) | 9 |
| _demean.py | demean, drop_singletons, absorbed_dof | 10 |
| _iv.py | iv2sls | 5 |
| _gmm.py | liml, gmm_iv | 5 |
| _panel.py | panel_fe | 5 |
| integration | - | 7 |
| **Total** | | **67** |

## Notes

- Plan: docs/plans/2026-03-05-polars-reg.md
- Architecture: Formula -> parse -> extract arrays -> demean (if FE) -> estimate -> SEs -> result
- LIML uses scipy.linalg.eig (not eigvalsh) to handle near-singular Y'MzY matrices
- CGM multi-way clustering supports arbitrary N dimensions via inclusion-exclusion
- Demeaning uses symmetric Kaczmarz + conjugate gradient acceleration (Correia 2016)
- DoF for multi-way FE uses pairwise connected components method
