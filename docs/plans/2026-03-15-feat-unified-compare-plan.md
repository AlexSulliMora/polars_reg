---
title: "feat: Unified compare() function with multi-backend support"
type: feat
date: 2026-03-15
---

# Unified compare() Function

## Overview

Replace `compare_stata()` and `compare_r()` with a single `compare()` function supporting backends: `stata`, `r`, `statsmodels`, `pyfixest`, `linearmodels`, and `all` (default). Each backend runs the equivalent regression and diffs coefficients/SEs against polars_reg.

**Breaking change**: `compare_stata()` and `compare_r()` removed from public API. `to_stata()` and `to_r()` kept for code generation.

See brainstorm: `docs/brainstorms/2026-03-15-unified-compare-brainstorm.md`

## Proposed Solution

```python
# Default: run all available backends
report = pr.compare("ols", "y ~ x1 + x2", df, vcov="HC1")

# Specific backend
report = pr.compare("ols", "y ~ x1", df, backend="pyfixest")

# Custom tolerance (default 1e-6, loosen for iterative estimators)
report = pr.compare("panel_fe", "y ~ x1", df, entity="firm", time="year", rtol=5e-2)
```

## Technical Approach

### Module Structure

New file: `polars_reg/_compare.py` containing:
- `compare()` — public function
- `ComparisonReport` — multi-backend result dataclass
- `BackendResult` — per-backend result dataclass
- Backend adapters: `_run_polars_reg()`, `_run_pyfixest()`, `_run_statsmodels()`, `_run_linearmodels()`, `_run_r()`, `_run_stata()`

Existing files kept (code generation only):
- `stata.py` — `to_stata()` stays, `compare_stata()` removed
- `r_equiv.py` — `to_r()` stays, `compare_r()` removed

### Backend Adapters

Each adapter follows the same contract:

```python
def _run_<backend>(
    estimator: str,
    formula: str,
    data: pl.DataFrame,
    vcov: str,
    cluster: list[str] | None,
    entity: str | None,
    time: str | None,
    **kwargs,
) -> BackendResult | None:
    """Run regression in <backend>. Returns None if backend unavailable or estimator unsupported."""
```

**Python backends** (in-process):
- Import the package inside the function (lazy import)
- Convert `pl.DataFrame` → `pd.DataFrame` for the backend call
- Fit the model, extract coefficients/SEs/N/R²
- Generate equivalent code string for the report
- Return `None` if package not installed (`ImportError`) or estimator not supported

**Stata/R backends** (subprocess):
- Reuse execution logic from `tests/stata_compare.py` and `tests/r_compare.py`
- Move the core execution helpers into `_compare.py` (or import from tests/)
- Return `None` if executable not found

### Estimator-to-Backend Translation

Each adapter maps polars_reg estimator + params to its package's API:

**pyfixest:**
| polars_reg | pyfixest call |
|-----------|---------------|
| `ols("y ~ x1")` | `pf.feols("y ~ x1", data=df)` |
| `ols("y ~ x1 \| fe1", cluster=["fe1"])` | `pf.feols("y ~ x1 \| fe1", data=df, vcov={"CRV1": "fe1"})` |
| `ols("y ~ x1", weights="w")` | `pf.feols("y ~ x1", data=df, weights="w")` |
| `iv2sls("y ~ x1 \|\| x_end ~ z1")` | `pf.feols("y ~ x1 \| x_end ~ z1", data=df)` |
| `panel_fe("y ~ x1", entity="firm")` | `pf.feols("y ~ x1 \| firm", data=df)` |
| `probit("y ~ x1")` | `pf.feglm("y ~ x1", data=df, family="probit")` |
| `logit("y ~ x1")` | `pf.feglm("y ~ x1", data=df, family="logit")` |
| `ppml("y ~ x1")` | `pf.fepois("y ~ x1", data=df)` |
| `quantreg("y ~ x1", tau=0.5)` | `pf.quantreg("y ~ x1", data=df, quantile=0.5)` |

**statsmodels:**
| polars_reg | statsmodels call |
|-----------|------------------|
| `ols("y ~ x1")` | `sm.OLS(y, X).fit()` |
| `ols("y ~ x1", vcov="HC1")` | `sm.OLS(y, X).fit(cov_type="HC1")` |
| `iv2sls("y ~ x1 \|\| x_end ~ z1")` | `IV2SLS(y, X_exog, X_endog, Z).fit()` |
| `probit("y ~ x1")` | `sm.Probit(y, X).fit()` |
| `logit("y ~ x1")` | `sm.Logit(y, X).fit()` |
| `ppml("y ~ x1")` | `sm.GLM(y, X, family=Poisson()).fit()` |
| `quantreg("y ~ x1", tau=0.5)` | `sm.QuantReg(y, X).fit(q=0.5)` |

**linearmodels:**
| polars_reg | linearmodels call |
|-----------|-------------------|
| `ols("y ~ x1 \| fe1")` | `AbsorbingLS(y, x, absorb=cats).fit()` |
| `iv2sls("y ~ x1 \|\| x_end ~ z1")` | `IV2SLS.from_formula("y ~ 1 + x1 + [x_end ~ z1]", data=df).fit()` |
| `liml(...)` | `IVLIML.from_formula("y ~ 1 + x1 + [x_end ~ z1]", data=df).fit()` |
| `gmm_iv(...)` | `IVGMM.from_formula("y ~ 1 + x1 + [x_end ~ z1]", data=df).fit()` |
| `panel_fe("y ~ x1", entity="firm")` | `PanelOLS.from_formula("y ~ x1 + EntityEffects", data=df).fit()` |
| `panel_re("y ~ x1", entity="firm")` | `RandomEffects.from_formula("y ~ 1 + x1", data=df).fit()` |
| `panel_fd("y ~ x1", entity="firm")` | `FirstDifferenceOLS.from_formula("y ~ x1", data=df).fit()` |

### Return Types

```python
@dataclass
class BackendResult:
    name: str                  # "pyfixest", "statsmodels", etc.
    coefs: NDArray
    se: NDArray
    names: list[str]
    n_obs: int
    r_squared: float | None
    code: str                  # equivalent code string
    max_coef_rdiff: float      # max relative diff vs polars_reg
    max_se_rdiff: float
    match: bool                # all within rtol

@dataclass
class ComparisonReport:
    estimator: str
    formula: str
    polars_coefs: NDArray
    polars_se: NDArray
    polars_names: list[str]
    polars_n_obs: int
    polars_r_squared: float
    backends: dict[str, BackendResult]
    skipped: dict[str, str]    # backend -> reason
    rtol: float

    def summary(self) -> str:
        """Print side-by-side comparison table."""

    def __repr__(self) -> str:
        """Short summary."""
```

## Implementation Plan

### Phase 1: Core Infrastructure

- [ ] Create `polars_reg/_compare.py` with `ComparisonReport`, `BackendResult` dataclasses
- [ ] Implement `_run_polars_reg()` — dispatch map (`{"ols": ols, "probit": probit, ...}`), run estimator, extract coefs/se/names/n_obs/r_squared
- [ ] Implement `compare()` function with `backend=` parameter dispatching
- [ ] Handle `backend="all"` — iterate all backends, collect results, skip unavailable

### Phase 2: Python Backend Adapters

- [ ] `_run_pyfixest()` — feols/feglm/fepois translation, vcov mapping, lazy import
- [ ] `_run_statsmodels()` — OLS/Probit/Logit/GLM/QuantReg/IV2SLS translation, lazy import
- [ ] `_run_linearmodels()` — PanelOLS/RandomEffects/IV2SLS translation, lazy import

### Phase 3: External Backend Adapters

- [ ] `_run_r()` — move R execution helpers from `tests/r_compare.py` into `_compare.py` (don't import from tests/), formula translation from `r_equiv.py`
- [ ] `_run_stata()` — move Stata execution helpers from `tests/stata_compare.py` into `_compare.py`, formula translation from `stata.py`

### Phase 4: Report and Display

- [ ] `ComparisonReport.summary()` — formatted multi-backend comparison table
- [ ] Coefficient name alignment across backends (handle `_cons` vs `(Intercept)` vs `Intercept`)
- [ ] Code string generation for each backend (included in report)

### Phase 5: Update Public API

- [ ] Add `compare` to `__init__.py` exports
- [ ] Remove `compare_stata` and `compare_r` from `__init__.py` exports
- [ ] Remove `compare_stata()` from `stata.py` and `compare_r()` from `r_equiv.py`
- [ ] Keep `to_stata()` and `to_r()` unchanged

### Phase 6: Tests

- [ ] Test `compare()` with `backend="pyfixest"` for OLS, OLS+FE, probit, logit, ppml
- [ ] Test `compare()` with `backend="statsmodels"` for OLS, probit, logit, ppml, quantreg
- [ ] Test `compare()` with `backend="linearmodels"` for panel_fe, panel_re, iv2sls
- [ ] Test `compare()` with `backend="r"` (skip if R not available)
- [ ] Test `compare()` with `backend="stata"` (skip if Stata not available)
- [ ] Test `compare()` with `backend="all"` — verify skipped backends noted
- [ ] Test unsupported estimator/backend combos return graceful skip
- [ ] Update `test_cross_parity.py` to use `compare()` instead of direct `assert_r_parity`/`assert_stata_parity`

### Phase 7: Documentation

- [ ] Update README: replace `compare_stata()`/`compare_r()` examples with `compare()`
- [ ] Update CLAUDE.md: note new public API
- [ ] Update `__init__.py` docstring

## Acceptance Criteria

- [ ] `pr.compare("ols", "y ~ x1", df)` runs all available backends and prints comparison
- [ ] `pr.compare("ols", "y ~ x1", df, backend="pyfixest")` runs only pyfixest
- [ ] Unavailable backends are skipped with a note, not an error
- [ ] `ComparisonReport.summary()` shows side-by-side coefficients/SEs with diffs
- [ ] Each `BackendResult.code` contains the equivalent code string
- [ ] Panel estimators map correctly to FE syntax in other packages
- [ ] Probit/logit/ppml/quantreg work with available backends
- [ ] `to_stata()` and `to_r()` still work unchanged
- [ ] All tests pass

## Backend Coverage (Verified via Documentation Research)

| polars_reg | pyfixest | statsmodels | linearmodels | R | Stata |
|-----------|----------|-------------|--------------|---|-------|
| `ols` | `feols()` | `OLS()` | `PooledOLS` | `lm`/`feols` | `reg`/`reghdfe` |
| `ols` + FE | `feols("y~x\|fe")` | - | `AbsorbingLS` / `PanelOLS` | `feols` | `reghdfe` |
| `ols` + WLS | `feols(weights=)` | `WLS()` | - | `lm(weights=)` | `reg [aw=]` |
| `iv2sls` | `feols("y~x\|fe\|end~iv")` | sandbox only | `IV2SLS` | `feols`/`ivreg` | `ivregress 2sls` |
| `liml` | - | - | `IVLIML` | `ivreg(method="LIML")` | - |
| `gmm_iv` | - | - | `IVGMM` | - | `ivregress gmm` |
| `panel_fe` | `feols("y~x\|entity")` | - | `PanelOLS(entity_effects)` | `plm(within)` | `xtreg,fe` |
| `panel_re` | - | - | `RandomEffects` | `plm(random)` | `xtreg,re` |
| `panel_fd` | - | - | `FirstDifferenceOLS` | `plm(fd)` | manual |
| `panel_ab` | - | - | - | - | `xtabond` |
| `panel_sys_gmm` | - | - | - | - | `xtdpdsys` |
| `probit` | `feglm(family="probit")` | `Probit()` | - | `glm(binomial/probit)` | `probit` |
| `logit` | `feglm(family="logit")` | `Logit()` | - | `glm(binomial/logit)` | `logit` |
| `ppml` | `fepois()` | `GLM(Poisson)` | - | `glm(poisson)` | `ppmlhdfe` |
| `quantreg` | `quantreg()` | `QuantReg()` | - | `quantreg::rq` | `qreg` |

`-` = not available in that backend.

### vcov Mapping by Backend

| polars_reg vcov | pyfixest | statsmodels | linearmodels | R (fixest) |
|----------------|----------|-------------|--------------|------------|
| `iid` | `"iid"` | `cov_type="nonrobust"` | `"unadjusted"` | `vcov="iid"` |
| `HC0` | - | `"HC0"` | - | - |
| `HC1` | `"hetero"` or `"HC1"` | `"HC1"` | `"robust"` | `vcov="hetero"` |
| `HC2` | `"HC2"` | `"HC2"` | - | `vcov="HC2"` |
| `HC3` | `"HC3"` | `"HC3"` | - | `vcov="HC3"` |
| `cluster` | `{"CRV1": "var"}` | `"cluster", groups=` | `"clustered", clusters=` | `vcov=~var` |
| `NW` | `"NW", lag=` | `"HAC", maxlags=` | `"kernel", bandwidth=` | `vcov="NW"` |
| `DK` | `"DK", time_id=` | `"HAC-Groupsum"` | `"kernel"` (panel) | - |

### Result Attribute Mapping

| polars_reg | pyfixest | statsmodels | linearmodels |
|-----------|----------|-------------|--------------|
| `.coefficients` | `.coef()` | `.params` | `.params` |
| `.se` | `.se()` | `.bse` | `.std_errors` |
| `.tstat` | `.tstat()` | `.tvalues` | `.tstats` |
| `.pvalue` | `.pvalue()` | `.pvalues` | `.pvalues` |
| `.n_obs` | `.nobs` | `.nobs` | `.nobs` |
| `.r_squared` | `.r2` | `.rsquared` | `.rsquared` |
| `._cons` name | `Intercept` | `const` | `const` |

## References

- Brainstorm: `docs/brainstorms/2026-03-15-unified-compare-brainstorm.md`
- Current Stata comparison: `polars_reg/stata.py`
- Current R comparison: `polars_reg/r_equiv.py`
- R test infrastructure: `tests/r_compare.py`
- Stata test infrastructure: `tests/stata_compare.py`
- Cross-parity tests: `tests/test_cross_parity.py`
