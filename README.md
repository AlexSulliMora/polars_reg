# polars_reg

Econometric regression methods using [Polars](https://pola.rs/) DataFrames.

## Features

- **OLS** with robust (HC0-HC3) and multi-way clustered standard errors
- **High-dimensional fixed effects** absorption (reghdfe-style iterative demeaning)
- **2SLS / IV** with first-stage F-statistics
- **LIML** (limited information maximum likelihood)
- **GMM-IV** with Hansen J overidentification test
- **Panel FE** (within estimator)

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

# Access results
result.coefficients  # coefficient vector
result.se            # standard errors
result.tstat         # t-statistics
result.pvalue        # p-values
result.confint()     # confidence intervals
result.r_squared     # R-squared
```

## Formula Syntax

| Formula | Stata equivalent |
|---------|-----------------|
| `y ~ x1 + x2` | `reg y x1 x2` |
| `y ~ x1 + x2 - 1` | `reg y x1 x2, noconstant` |
| `y ~ x1 \| fe1 + fe2` | `reghdfe y x1, absorb(fe1 fe2)` |
| `y ~ x_exog \|\| x_endog ~ z1 + z2` | `ivregress 2sls y x_exog (x_endog = z1 z2)` |

## Estimators

| Function | Description |
|----------|-------------|
| `ols()` | OLS with optional FE absorption |
| `iv2sls()` | Two-stage least squares |
| `liml()` | Limited information maximum likelihood |
| `gmm_iv()` | Two-step efficient GMM |
| `panel_fe()` | Panel fixed effects (within) |

## Standard Error Options

- `vcov="iid"` — homoskedastic (default)
- `vcov="HC1"` — heteroskedasticity-robust (Stata's `robust`)
- `vcov="HC0"`, `"HC2"`, `"HC3"` — other HC variants
- `cluster=["firm_id"]` — one-way clustered
- `cluster=["firm_id", "year_id"]` — two-way clustered (Cameron-Gelbach-Miller)

## Requirements

- Python >= 3.11
- Polars >= 1.0
- NumPy >= 1.24
- SciPy >= 1.10
