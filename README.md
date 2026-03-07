# polars_reg

Econometric regression methods using [Polars](https://pola.rs/) DataFrames. Also accepts pandas DataFrames.

## Features

- **OLS** with robust (HC0-HC3) and multi-way clustered standard errors
- **High-dimensional fixed effects** absorption (reghdfe-style iterative demeaning)
- **Weighted Least Squares** — analytic weights (`weights=`) and frequency weights (`fweights=`)
- **2SLS / IV** with first-stage F-statistics and weak instrument diagnostics
- **LIML** (limited information maximum likelihood)
- **GMM-IV** with Hansen J overidentification test
- **Panel estimators**: fixed effects (within), random effects (Swamy-Arora GLS), first-difference
- **Dynamic panel GMM**: Arellano-Bond (difference GMM) and Blundell-Bond (system GMM)
- **Probit / Logit** with MLE, marginal effects, and odds ratios
- **Quantile regression** — median and arbitrary quantiles with bootstrap SEs
- **Bootstrap SEs** — pairs bootstrap and wild cluster bootstrap (Webb 6-point)
- **HAC / Driscoll-Kraay** standard errors for time series and panel data
- **GroupBy regression**: run the same regression per group (e.g., per stock, per industry)
- **regtable**: side-by-side regression comparison tables (estout/esttab-style), with LaTeX and HTML export
- **Diagnostics**: Wald test, Hausman test (FE vs RE), Kleibergen-Paap, Stock-Yogo weak IV
- **Stata/R equivalence**: generate equivalent Stata or R code for any specification

All estimators are validated against Stata output to 5+ decimal places.

## Installation

```bash
pip install polars_reg
```

## Quick Start

```python
import polars as pl
import polars_reg as pr

df = pl.read_csv("data.csv")

# OLS with robust standard errors
result = pr.ols("y ~ x1 + x2 + x3", data=df, vcov="HC1")
print(result.summary())

# OLS with absorbed fixed effects and clustered SEs (like Stata's reghdfe)
result = pr.ols("y ~ x1 + x2 | firm_id + year_id", data=df, cluster=["firm_id"])

# IV / 2SLS
result = pr.iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=df)

# Panel fixed effects
result = pr.panel_fe("y ~ x1 + x2", data=df, entity="firm_id", time="year_id")

# GroupBy: run regression per industry
grp = pr.groupby_reg(pr.ols, "y ~ x1 + x2", df, group_by="industry")
grp.coef_table()  # stacked Polars DataFrame

# Side-by-side comparison table
pr.regtable(m1, m2, m3, labels=["OLS", "Robust", "FE"])

# Probit / Logit
result = pr.logit("y_binary ~ x1 + x2", data=df, cluster="firm_id")
pr.marginal_effects(result, at="mean")
pr.odds_ratios(result)

# Quantile regression (median)
result = pr.quantreg("y ~ x1 + x2", data=df, tau=0.5)

# Dynamic panel GMM
result = pr.panel_ab("y ~ x1", data=df, entity="firm_id", time="year_id")
result = pr.panel_sys_gmm("y ~ x1", data=df, entity="firm_id", time="year_id")

# Access results
result.coefficients  # coefficient vector
result.se            # standard errors
result.tstat         # t-statistics
result.pvalue        # p-values
result.confint()     # confidence intervals
result.coef_table()  # Polars DataFrame
result.wald_test(R)  # Wald test for linear restrictions
```

## Formula Syntax

| Formula | Meaning | Stata | R |
|---------|---------|-------|---|
| `y ~ x1 + x2` | OLS | `reg y x1 x2` | `lm(y ~ x1 + x2)` |
| `y ~ x1 + x2 - 1` | No intercept | `reg y x1 x2, noconstant` | `lm(y ~ x1 + x2 - 1)` |
| `y ~ x1 \| fe1 + fe2` | Absorbed FE | `reghdfe y x1, absorb(fe1 fe2)` | `feols(y ~ x1 \| fe1 + fe2)` |
| `y ~ x1 \|\| x_end ~ z1 + z2` | IV/2SLS | `ivregress 2sls y x1 (x_end = z1 z2)` | `feols(y ~ x1 \| 0 \| x_end ~ z1 + z2)` |
| `y ~ x1 \| fe1 \| x_end ~ z1` | IV + FE | `ivreghdfe y x1 (x_end = z1), absorb(fe1)` | `feols(y ~ x1 \| fe1 \| x_end ~ z1)` |
| `y ~ x1*x2` | Full factorial | `reg y c.x1##c.x2` | `lm(y ~ x1 * x2)` |
| `y ~ x1:x2` | Interaction only | `reg y c.x1#c.x2` | `lm(y ~ x1:x2)` |
| `y ~ i.group + x1` | Indicator dummies | `reg y i.group x1` | `lm(y ~ factor(group) + x1)` |
| `y ~ i.group*x1` | Indicator × continuous | `reg y i.group#c.x1` | `lm(y ~ factor(group) * x1)` |

## Estimators

| Function | Description |
|----------|-------------|
| `ols()` | OLS/WLS with optional FE absorption |
| `iv2sls()` | Two-stage least squares |
| `liml()` | Limited information maximum likelihood |
| `gmm_iv()` | Two-step efficient GMM |
| `panel_fe()` | Panel fixed effects (within) |
| `panel_re()` | Panel random effects (Swamy-Arora GLS) |
| `panel_fd()` | Panel first-difference |
| `panel_ab()` | Arellano-Bond dynamic panel GMM |
| `panel_sys_gmm()` | Blundell-Bond system GMM |
| `probit()` | Probit MLE |
| `logit()` | Logit MLE |
| `quantreg()` | Quantile regression (IRLS + bootstrap) |
| `groupby_reg()` | Run any estimator per group |
| `regtable()` | Side-by-side regression table |
| `marginal_effects()` | Probit/logit marginal effects |
| `odds_ratios()` | Logit odds ratios with delta-method SEs |
| `hausman_test()` | Hausman specification test (FE vs RE) |

## Standard Error Options

- `vcov="iid"` — homoskedastic (default)
- `vcov="HC1"` — heteroskedasticity-robust (Stata's `robust`)
- `vcov="HC0"`, `"HC2"`, `"HC3"` — other HC variants
- `cluster=["firm_id"]` — one-way clustered
- `cluster=["firm_id", "year_id"]` — two-way clustered (Cameron-Gelbach-Miller)
- `vcov="NW"` — Newey-West HAC (requires `time=`)
- `vcov="DK"` — Driscoll-Kraay (requires `time=`)
- `vcov="bootstrap"` — pairs bootstrap
- `vcov="wildboot"` — wild cluster bootstrap (requires `cluster=`)

## Stata / R Equivalence

Generate equivalent code to verify results in Stata or R:

```python
# Stata code
print(pr.to_stata("ols", "y ~ x1 + x2 | firm_id", cluster=["firm_id"]))
# → reghdfe y x1 x2, absorb(firm_id) vce(cluster firm_id)

# R code
print(pr.to_r("ols", "y ~ x1 + x2 | firm_id", cluster=["firm_id"]))
# → library(fixest)
#   model <- feols(y ~ x1 + x2 | firm_id, data=df, vcov=~firm_id)
```

## Showcase

See the [showcase notebook](notebooks/showcase.ipynb) for a full tour of all features, or view the [rendered PDF](notebooks/showcase.pdf) with all outputs included.

## Requirements

- Python >= 3.11
- Polars >= 1.0
- NumPy >= 1.24
- SciPy >= 1.10
- pandas (optional — for pandas DataFrame input)
