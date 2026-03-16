# Brainstorm: Unified compare() Function

**Date:** 2026-03-15

## What We're Building

A single `compare()` function replacing `compare_stata()` and `compare_r()`, with support for Python backends (statsmodels, pyfixest, linearmodels) alongside Stata and R.

```python
# Default: run all available backends
report = pr.compare("ols", "y ~ x1 + x2", df)

# Specific backend
report = pr.compare("ols", "y ~ x1 + x2", df, backend="pyfixest")

# Code generation kept separate
pr.to_stata("ols", "y ~ x1 + x2")
pr.to_r("ols", "y ~ x1 + x2")
```

## Key Decisions

1. **Replace, don't wrap** — `compare_stata()` and `compare_r()` removed. Single `compare()` with `backend=` param. Breaking change (already at 0.2.0).
2. **`backend="all"` is the default** — attempts every available backend, skips unavailable ones with a note in the report (no error, no warning).
3. **Python backends run in-process** — import the package, fit the model, extract results directly. No subprocess, no exec(). Graceful skip if package not installed.
4. **Stata/R backends use existing subprocess infrastructure** — reuse `tests/stata_compare.py` and `tests/r_compare.py` execution logic.
5. **Report includes equivalent code** — `ComparisonReport.code` shows what was run in each backend. Useful for reproducibility.
6. **Panel estimators are replicable via OLS+FE** — `panel_fe("y ~ x1", entity="firm")` maps to `feols("y ~ x1 | firm")` in pyfixest, `xtreg y x1, fe` in Stata, etc.

## Backend Coverage Matrix

| Estimator | polars_reg | pyfixest | statsmodels | linearmodels | R | Stata |
|-----------|-----------|----------|-------------|--------------|---|-------|
| ols | ols() | feols() | OLS() | - | lm/feols | reg/reghdfe |
| ols + FE | ols() | feols() | - | PanelOLS | feols | reghdfe |
| iv2sls | iv2sls() | feols() | IV2SLS() | IV2SLS() | feols/ivreg | ivregress |
| liml | liml() | - | - | - | ivreg(LIML) | - |
| gmm_iv | gmm_iv() | - | - | - | - | ivregress gmm |
| panel_fe | panel_fe() | feols(+FE) | - | PanelOLS | plm(within) | xtreg,fe |
| panel_re | panel_re() | - | - | RandomEffects | plm(random) | xtreg,re |
| panel_fd | panel_fd() | - | - | FirstDiffOLS | plm(fd) | reg (manual) |
| panel_ab | panel_ab() | - | - | ? | - | xtabond |
| panel_sys_gmm | panel_sys_gmm() | - | - | ? | - | xtdpdsys |
| probit | probit() | feglm(probit) | Probit() | - | glm(binomial) | probit |
| logit | logit() | feglm(logit) | Logit() | - | glm(binomial) | logit |
| ppml | ppml() | fepois() | GLM(Poisson) | - | glm(poisson) | ppmlhdfe |
| quantreg | quantreg() | - | QuantReg() | - | quantreg::rq | qreg |

## Report Format

```
compare("ols", "y ~ x1 + x2 | firm_id", df, cluster=["firm_id"])

                  polars_reg    pyfixest   statsmodels        R       Stata
─────────────────────────────────────────────────────────────────────────────
x1                  1.2345       1.2345      1.2345       1.2345     1.2345
  (SE)             (0.0456)     (0.0456)    (0.0456)     (0.0456)   (0.0456)
x2                  0.5678       0.5678      0.5678       0.5678     0.5678
  (SE)             (0.0234)     (0.0234)    (0.0234)     (0.0234)   (0.0234)
─────────────────────────────────────────────────────────────────────────────
N                    1000         1000        1000         1000       1000
R²                  0.4567       0.4567      0.4567       0.4567     0.4567
─────────────────────────────────────────────────────────────────────────────
Max |Δcoef|                     1.2e-10     3.4e-08      2.1e-06    1.5e-06
Max |Δse|                       2.3e-10     5.6e-08      4.2e-06    3.1e-06
Match (rtol=1e-6)                 ✓            ✓            ✓          ✓

Skipped: linearmodels (not installed)
```

## Parameter Forwarding

`compare()` accepts the same parameters as the polars_reg estimator, plus `backend=`:

```python
# compare() signature mirrors the estimator's params
pr.compare("ols", "y ~ x1 + x2", df, vcov="HC1", cluster=["firm"], backend="all")
pr.compare("panel_fe", "y ~ x1", df, entity="firm", time="year", backend="pyfixest")
pr.compare("probit", "y ~ x1", df, backend="statsmodels")
```

Internally: polars_reg params (`vcov`, `cluster`, `entity`, `time`, etc.) are forwarded to the polars_reg estimator call. Each backend adapter translates these into its own calling convention. Unknown params for a backend are silently ignored (e.g., pyfixest doesn't need `entity=` because it uses formula FE syntax).

## Return Type

```python
@dataclass
class ComparisonReport:
    estimator: str
    formula: str
    polars_result: RegressionResult
    backends: dict[str, BackendResult]  # backend_name -> result
    skipped: dict[str, str]             # backend_name -> reason

@dataclass
class BackendResult:
    coefs: NDArray
    se: NDArray
    names: list[str]
    n_obs: int
    r_squared: float | None
    code: str              # equivalent code string
    max_coef_rdiff: float  # vs polars_reg
    max_se_rdiff: float
    match: bool            # within rtol
```

## Open Questions

- **linearmodels panel_ab/panel_sys_gmm**: unknown if `FirstDifferenceGMM` or similar classes match our implementation. Investigate during implementation; mark as unsupported if no clean mapping.
- **Quantile regression in pyfixest**: pyfixest may not support quantile regression. Verify during implementation.

## Scope

- New module: `polars_reg/_compare.py` with `compare()` function
- Backend adapters: one function per backend that translates formula + runs + extracts results
- Reuse existing `stata.py` and `r_equiv.py` for code generation (`to_stata`, `to_r` stay)
- Move comparison execution logic from `stata.py`/`r_equiv.py` into `_compare.py`
- Update `__init__.py`: export `compare`, remove `compare_stata`/`compare_r`
- Update tests
