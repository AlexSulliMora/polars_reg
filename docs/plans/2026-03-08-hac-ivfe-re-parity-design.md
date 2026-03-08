# HAC/DK SEs, IV+FE Tests, RE GLS, Stata Parity — Design

**Goal:** Four features in order: (1) HAC/Driscoll-Kraay SEs for all estimators with Rust acceleration, (2) IV with absorbed FE test coverage, (3) Random effects GLS with full SE support, (4) Stata parity validation via frozen fixtures.

---

## Feature 1: HAC / Driscoll-Kraay SEs for all estimators

### Problem

`vcov_hac()` and `vcov_driscoll_kraay()` exist in `_se.py` and work for OLS and `panel_fe()`. They are missing from `iv2sls()`, `liml()`, and `gmm_iv()`. Additionally, the meat computation is pure NumPy — worth moving to Rust for large N.

### Design

The existing HAC/DK functions compute the bread `(X'X)^{-1}` internally. IV estimators use different bread matrices:
- **2SLS**: `(X_hat'X)^{-1}`
- **LIML**: `(X_w'X_full)^{-1}` (LIML-weighted regressors)
- **GMM**: `(X'Z S^{-1} Z'X)^{-1}`

**Solution**: Factor out meat computation from bread.

**`_se.py` changes:**
- Extract `_hac_meat(scores, time_ids, bandwidth)` — computes Newey-West kernel `Γ₀ + Σⱼ w(j)(Γⱼ + Γⱼ')` on score matrix
- Extract `_dk_meat(scores, time_ids, bandwidth)` — aggregates scores by time, then applies NW kernel
- Existing `vcov_hac()` / `vcov_driscoll_kraay()` become thin wrappers: `bread @ meat @ bread`

**Rust (`src/lib.rs`) changes:**
- Add `rust_hac_meat(scores: 2D array, time_ids: 1D array, bandwidth: int)` → k×k meat matrix
- Add `rust_dk_meat(scores: 2D array, time_ids: 1D array, bandwidth: int)` → k×k meat matrix
- O(T·k²·L) kernel loop, parallelizable with rayon over lags
- Python `_hac_meat()` / `_dk_meat()` call Rust when available, fall back to NumPy

**IV estimator changes (`_iv.py`, `_gmm.py`):**
- Add `time: str | None = None` and `bandwidth: int | None = None` parameters to `iv2sls()`, `liml()`, `gmm_iv()`
- Add `"NW"` and `"DK"` to accepted vcov types
- Route through meat helpers with estimator-specific bread and score matrices:
  - 2SLS: scores from `X_hat`, bread `(X_hat'X)^{-1}`
  - LIML: scores from `X_w`, bread `(X_w'X_full)^{-1}`
  - GMM: effective scores from moment conditions, GMM bread

**Bandwidth**: Default `floor(4*(T/100)^(2/9))` matching Stata `newey`. User-overridable via `bw=` parameter.

**Rust fast paths** (`rust_ols_nofe`, `rust_ols_from_arrays`, `rust_iv2sls`) don't change — HAC/DK falls back to Python solve + Rust meat.

---

## Feature 2: IV with absorbed FE — test coverage

### Problem

IV + FE works (both Python and Rust paths demean all variables before 2SLS) but has zero test coverage.

### Tests needed

- Basic correctness: IV + 1-way FE, IV + 2-way FE — verify coefficients match manual demean-then-2SLS
- SE types with FE: iid, HC0/HC1, clustered, multi-way clustered, NW, DK
- First-stage F-stat with FE: verify computed correctly after demeaning
- Singleton dropping: FE groups with single observation dropped
- DoF adjustment: `df_abs` from absorbed FE flows through to SE computation
- Rust vs Python parity: `_iv2sls_rust()` and Python fallback give identical results
- Edge cases: overidentified IV + FE, exactly-identified IV + FE, FE column same as cluster column

---

## Feature 3: Random Effects GLS — full SE support

### Problem

`panel_re()` uses Swamy-Arora variance component estimation with quasi-demeaning (θ). Only supports `vcov="iid"` and `"bootstrap"`. Needs full SE support matching `panel_fe()`.

### Design

**`_panel.py` changes:**
- Add `time: str | None = None` and `bandwidth: int | None = None` parameters to `panel_re()`
- After GLS solve, compute residuals and route through standard SE functions:
  - **Clustered**: cluster on entity (default) or user-specified. Quasi-demeaned X as regressor in sandwich.
  - **HC0/HC1**: robust sandwich with quasi-demeaned X and residuals
  - **NW/DK**: requires `time` parameter, uses Rust-accelerated meat helpers from Feature 1
  - **Wildboot**: wild cluster bootstrap on quasi-demeaned model

**Bread**: `(X̃'X̃)^{-1}` where X̃ is quasi-demeaned X (already computed).

**Residuals for sandwich**: From original model `e = y - Xβ`, not transformed model. Matches Stata `xtreg, re` behavior. (To be verified against Stata output in Feature 4.)

**No Rust changes**: panel_re is low-frequency; Python SE computation is fine. Rust meat helpers from Feature 1 used for NW/DK.

---

## Feature 4: Stata parity validation — frozen fixtures

### Problem

No ground-truth validation against Stata. Test infrastructure exists in `test_stata_parity.py` but needs actual fixture data.

### Design

**Approach**: Frozen fixtures — run Stata once locally, commit results, test against them in CI.

**Dataset**: Single synthetic dataset (N=10,000, ~100 firms, ~20 years, 2 exog, 1 endog, 2 instruments), fixed seed. Saved as `tests/fixtures/parity_data.csv`.

**Fixture structure**:
```
tests/fixtures/
    parity_data.csv
    stata/
        generate_fixtures.do     # Stata script to regenerate all fixtures
        ols_iid.csv
        ols_hc1.csv
        ols_hc2.csv
        ols_hc3.csv
        ols_cluster.csv
        ols_nw.csv
        ols_dk.csv
        ols_fe_cluster.csv
        ols_fe_hc2.csv
        ols_2fe_cluster.csv
        iv_iid.csv
        iv_robust.csv
        iv_cluster.csv
        iv_fe_cluster.csv
        re_iid.csv
        re_cluster.csv
        newey.csv
```

**Each CSV contains**: variable names, coefficients, standard errors, t-stats, p-values, R², N, and estimator-specific stats (first-stage F, Hansen J, etc.)

**Tolerance**: 6 decimal places for coefficients, 4 for SEs.

**Test file**: `tests/test_stata_parity.py` — parametrized pytest loads each fixture, runs equivalent polars_reg call, compares. Runs in normal CI (no Stata needed).

**Stata `.do` file**: `tests/fixtures/stata/generate_fixtures.do` — self-contained script using `reghdfe`, `ivreghdfe`, `xtreg`, `newey`. User reruns locally if fixtures need updating.

---

## Implementation Order

1. HAC/DK SEs (refactor _se.py, Rust meat, extend IV/GMM)
2. IV + FE test coverage
3. RE GLS full SE support
4. Stata parity fixtures (big batch at the end)
