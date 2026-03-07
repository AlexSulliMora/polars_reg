# polars_reg

Econometric regression methods using [Polars](https://pola.rs/) DataFrames. Also accepts pandas DataFrames.

## Features

- **OLS** with robust (HC0-HC3) and multi-way clustered standard errors
- **High-dimensional fixed effects** absorption (reghdfe-style iterative demeaning)
- **2SLS / IV** with first-stage F-statistics
- **LIML** (limited information maximum likelihood)
- **GMM-IV** with Hansen J overidentification test
- **Panel estimators**: fixed effects (within), random effects (Swamy-Arora GLS), first-difference
- **GroupBy regression**: run the same regression per group (e.g., per stock, per industry)
- **regtable**: side-by-side regression comparison tables (estout/esttab-style)
- **Diagnostics**: Wald test, Hausman test (FE vs RE)
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

## Estimators

| Function | Description |
|----------|-------------|
| `ols()` | OLS with optional FE absorption |
| `iv2sls()` | Two-stage least squares |
| `liml()` | Limited information maximum likelihood |
| `gmm_iv()` | Two-step efficient GMM |
| `panel_fe()` | Panel fixed effects (within) |
| `panel_re()` | Panel random effects (Swamy-Arora GLS) |
| `panel_fd()` | Panel first-difference |
| `groupby_reg()` | Run any estimator per group |
| `regtable()` | Side-by-side regression table |
| `hausman_test()` | Hausman specification test (FE vs RE) |

## Standard Error Options

- `vcov="iid"` — homoskedastic (default)
- `vcov="HC1"` — heteroskedasticity-robust (Stata's `robust`)
- `vcov="HC0"`, `"HC2"`, `"HC3"` — other HC variants
- `cluster=["firm_id"]` — one-way clustered
- `cluster=["firm_id", "year_id"]` — two-way clustered (Cameron-Gelbach-Miller)

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

## Showcase Notebook

See [`notebooks/showcase.ipynb`](notebooks/showcase.ipynb) for a full tour of all features.

## Requirements

- Python >= 3.11
- Polars >= 1.0
- NumPy >= 1.24
- SciPy >= 1.10
- pandas (optional — for pandas DataFrame input)
