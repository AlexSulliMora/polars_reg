# API Consistency Principles

How public functions should look: return types, parameter naming, error handling, and integration contracts.

## Return Type Contract

**All estimator functions return `RegressionResult`.** This is what enables `group_by_reg()` and `regtable()` to work with any estimator.

**New diagnostic functions** should return a typed dataclass with a `.summary()` method, following `GRSTestResult` in `_diagnostics.py`:

```python
@dataclass
class GRSTestResult:
    """Result of a GRS (Gibbons-Ross-Shanken 1989) F-test."""
    f_stat: float
    p_value: float
    ...
    def summary(self) -> str: ...
```

**Legacy diagnostics** (`hausman_test()`, `weak_instrument_test()`, `kleibergen_paap_test()`) return plain `dict`. These are left as-is -- noted as tech debt.

**Avoid union return types.** `quantreg()` returns `RegressionResult | list[RegressionResult]` depending on whether `tau` is scalar or list. New estimators should not follow this pattern -- prefer always returning a single type.

**RegressionResult extension fields:** some estimators attach private fields via `setattr()` (e.g., `result._X`, `result._iv_X_exog`). New estimator-specific fields should be declared on the dataclass with `Optional` typing rather than using ad-hoc `setattr()`.

## Parameter Naming

Canonical parameter names and their types:

| Parameter  | Type                           | Semantics                                          |
|------------|--------------------------------|----------------------------------------------------|
| `formula`  | `str`                          | Wilkinson notation: `"y ~ x1 + x2 \| fe1 + fe2"`  |
| `data`     | `pl.DataFrame \| pl.LazyFrame` | Input data (Polars only, no pandas)                |
| `entity`   | `str`                          | Panel entity column name                           |
| `time`     | `str \| None`                  | Panel time column / HAC ordering column            |
| `vcov`     | `str`                          | One of the vcov vocabulary strings (see below)     |
| `cluster`  | `str \| list[str] \| None`     | Column name(s) for clustered SEs                   |
| `ssc`      | `SSC \| None`                  | Small-sample correction configuration              |
| `bandwidth`| `int \| None`                  | Number of lags for HAC/DK                          |
| `weights`  | `str \| None`                  | Column name for analytic weights                   |
| `fweights` | `str \| None`                  | Column name for frequency weights                  |
| `n_boot`   | `int`                          | Number of bootstrap replications                   |
| `seed`     | `int \| None`                  | RNG seed for reproducibility                       |

**Never use:** `df`, `fml`, `cl`, `se_type`, `B`, `panel_id`.

**Dual semantics of `time`:** in panel estimators, `time` is a structural panel dimension (column name). In cross-sectional estimators with HAC/DK, `time` is a vcov-related ordering column. These are semantically different despite sharing a name.

## Parameter Ordering

For **estimator functions only**, parameters follow this order:

```
(formula, data, [entity, time], vcov, cluster, ssc, [time, bandwidth], [weights, fweights], [n_boot, seed])
```

- `formula` and `data` are always first and second
- Estimator-specific required params (`entity`, `time` for panel; `tau` for quantile; `lags` for Arellano-Bond) come after `data` but before `vcov`
- `vcov` and `cluster` are the first optional/keyword params
- `ssc` follows `cluster` — controls the degrees-of-freedom correction applied to each vcov type
- HAC/DK params (`time`, `bandwidth`) follow `ssc`
- Weight params follow HAC
- Bootstrap params are last

**Example:** `ols()` signature at `_ols.py:371`:

```python
def ols(formula, data, vcov="iid", cluster=None, time=None, bandwidth=None,
        weights=None, fweights=None, n_boot=999, seed=None)
```

**Non-estimator functions** (diagnostics, utilities) follow "most important argument first, keyword-only for optionals" with no strict ordering template.

**Exceptions:** `panel_ab()` and `panel_sys_gmm()` have no `vcov`/`cluster` parameters -- VCV is determined by the estimation method (robust by construction).

## vcov Vocabulary

Input vocabulary:

```python
{"iid", "HC0", "HC1", "HC2", "HC3", "NW", "DK", "bootstrap", "wildboot"}
```

`"robust"` is **not** a valid input value. Stata's `robust` maps to HC1, but R's `sandwich` package defaults to HC0 — the ambiguity invites mistakes. Use the explicit HC variant instead.

**SSC interaction:** The `ssc` parameter (`SSC | None`) controls the degrees-of-freedom correction applied to each vcov type. The default SSC (`k_fixef="nonnested", G_df="min"`) matches pyfixest, Stata `reghdfe`, and R `fixest`. Different backends may use different SSC conventions for specific estimators (e.g., Stata `ivregress` uses `k_adj=False, G_adj=False`). The `_backend_ssc()` function in `_ssc.py` maps backend names to their SSC conventions, and `compare(match_ssc=True)` uses this to run polars_reg with each backend's SSC for apples-to-apples SE comparison.

**Minimum set for new estimators:**
- *vcov strings:* `{iid, HC1}`
- *Functionality:* iid + HC1 + one-way clustered (clustering uses the `cluster` parameter)
- Bootstrap and HAC are optional

**Unsupported types:** raise `ValueError` listing available options:

```python
raise ValueError(
    f"vcov={vcov!r} is not supported for {model_type}. "
    f"Available: {', '.join(sorted(supported))}"
)
```

**Default:** `"iid"` for most estimators. MLE-based estimators may default to `"HC1"` when the statistical theory requires robust SEs (e.g., PPML is quasi-MLE, so sandwich VCV is standard).

**vcov support by estimator:**

| Estimator    | iid | HC0-3 | NW | DK | bootstrap | wildboot | cluster |
|--------------|-----|-------|----|----|-----------|----------|---------|
| `ols`        | Y   | Y     | Y  | Y  | Y         | Y        | Y       |
| `iv2sls`     | Y   | HC0-1 | Y  | Y  | Y         | Y        | Y       |
| `liml`       | Y   | HC0-1 | Y  | Y  | Y         | Y        | Y       |
| `gmm_iv`     | Y   | HC0-1 | Y  | Y  | Y         | Y        | Y       |
| `panel_fe`   | Y   | -     | Y  | Y  | Y         | Y        | Y       |
| `panel_re`   | Y   | -     | Y  | -  | Y         | Y        | Y       |
| `panel_fd`   | Y   | HC1   | -  | -  | Y         | Y        | Y       |
| `probit`     | Y   | HC1   | -  | -  | -         | -        | Y       |
| `logit`      | Y   | HC1   | -  | -  | -         | -        | Y       |
| `ppml`       | -   | HC1*  | -  | -  | -         | -        | Y       |
| `quantreg`   | -   | -     | -  | -  | boot only | -        | -       |
| `panel_ab`   | -   | -     | -  | -  | -         | -        | -       |
| `panel_sys_gmm` | -| -     | -  | -  | -         | -        | -       |

\* PPML defaults to HC1; iid is not offered because sandwich VCV is theoretically required.

## model_type Vocabulary

Current values assigned to `RegressionResult.model_type`:

| Value              | Set by         | Notes                        |
|--------------------|----------------|------------------------------|
| `"OLS"`            | `ols()`        |                              |
| `"WLS"`            | `ols()`        | When `weights` provided      |
| `"OLS (fweight)"`  | `ols()`        | When `fweights` provided     |
| `"2SLS"`           | `iv2sls()`     |                              |
| `"LIML"`           | `liml()`       |                              |
| `"GMM"`            | `gmm_iv()`     |                              |
| `"Panel FE"`       | `panel_fe()`   |                              |
| `"Panel RE"`       | `panel_re()`   |                              |
| `"Panel FD"`       | `panel_fd()`   |                              |
| `"Arellano-Bond"`  | `panel_ab()`   |                              |
| `"System GMM"`     | `panel_sys_gmm()` |                           |
| `"Probit"`         | `probit()`     |                              |
| `"Logit"`          | `logit()`      |                              |
| `"Quantile(τ)"`    | `quantreg()`   | Parameterized (e.g., `"Quantile(0.50)"`) |
| `"PPML"`           | `ppml()`       |                              |

**Naming convention:** all-caps for acronyms (`"OLS"`, `"2SLS"`, `"PPML"`), title-case for descriptive names (`"Panel FE"`, `"Arellano-Bond"`), title-case for proper nouns (`"Probit"`, `"Logit"`).

`"Quantile(τ)"` is the only parameterized model_type -- it varies at runtime. This is a sanctioned exception.

## Error Handling Boundaries

| Exception             | Use when                                                        | Examples                                          |
|-----------------------|-----------------------------------------------------------------|---------------------------------------------------|
| `ValueError`          | Data-integrity issues, invalid arguments                        | Singular matrix, no observations, unsupported vcov |
| `TypeError`           | Wrong argument types                                            | String passed where DataFrame expected            |
| `NotImplementedError` | Feature combinations not yet supported                          | FE absorption in LIML, multi-way cluster in GMM   |
| `RuntimeError`        | Infrastructure failures                                         | Missing native extension                          |
| `warnings.warn()`     | Statistical judgment calls                                      | Low power, non-convergence, PPML separation       |

**LinAlgError policy:** linear algebra exceptions from NumPy/SciPy should be caught at the estimator layer and re-raised as `ValueError` with a descriptive message. Never let raw `np.linalg.LinAlgError` propagate to the user.

**Error message pattern:** include (1) the problematic value, (2) what's wrong, (3) available options:

```python
raise ValueError(
    f"vcov={vcov!r} is not supported for {model_type}. "
    f"Available: {', '.join(sorted(supported))}"
)
```

### Philosophy

From `docs/solutions/runtime-errors/polars-reg-comprehensive-code-review.md`:

> "Silent wrong answers are worse than crashes."

A crash is a gift -- it tells you something is wrong. Silent corruption (wrong DoF, misaligned weights, NaN propagation) is the real enemy. Prefer raising over silently computing wrong results.

## Composability Contract

Any estimator that meets these four conditions works automatically with `group_by_reg()` and `regtable()`:

1. Accepts `formula` as the first positional argument
2. Accepts `data` as a keyword argument
3. Returns `RegressionResult`
4. Populates the standard fields: `names`, `coefficients`, `vcov`, `n_obs`, `r_squared`, `model_type`

`regtable()` renders from: `params`, `se`, `pvalues`, `nobs`, `r_squared`, `model_type`.

## Output Formatting

- `.summary()`: coefficient table with Coef / SE / t / P>|t| / [95% CI] columns
- `regtable()`: handles significance stars and multi-model comparison
- `GRSTestResult.summary()` is the template for new diagnostic summaries

**Gold-standard docstring:** `_ppml.py:46-73` -- has summary, extended description, full journal citation in `Reference:` line, complete `Args:` with types and defaults, and `Returns:` block.

## Naming Conventions

**Function names:** use underscores between all words. Never concatenate words.
- Correct: `group_by_reg`, `rolling_reg`, `fama_macbeth`, `marginal_effects`
- Wrong: `groupbyreg`, `rollingreg`, `famaMacBeth`

**Parameter names:** same rule — underscores between all words.
- Correct: `new_data`, `group_by`, `min_obs`
- Wrong: `newdata`, `groupby`, `minobs`

**Class names:** PascalCase (standard Python). Acronyms stay uppercase: `SSC`, `RegressionResult`, `GroupRegressionResult`.

**Internal functions/methods:** underscore prefix + same word-separation rule: `_build_new_data_X`, `_first_difference`.

**model_type strings:** see model_type Vocabulary section (all-caps for acronyms, title-case for descriptive).

## Type Annotations

- Public functions: full annotations on all parameters and return type
- Internal functions: annotate parameters and return types
- Stable dict returns should use `TypedDict` where the key set is fixed
