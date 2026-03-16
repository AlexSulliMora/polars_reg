# Statistical Rigor Principles

How calculations are justified, cited, and validated against reference implementations.

## Citation Standards

### When to Cite

Cite formulas where the implementation choice affects numerical results versus a reference implementation (Stata/R). Standard linear algebra identities (e.g., `beta = (X'X)^{-1}X'y`) don't need citations.

**Current baseline:** near zero equation-level citations exist in the codebase. Only `Kamstra & Shi (2021), eq. 7` in `_diagnostics.py:395` and `Stock & Yogo (2005), Table 5.2` in `_diagnostics.py:14` have equation/table-level refs. New code must follow the standard below; retroactive coverage is a separate, lower-priority task.

### Citation Format

**In code comments** -- equation/table/section-level reference next to the implementing line:

```python
# Cameron, Gelbach & Miller (2011), eq 2.3
dfc = n_groups / (n_groups - 1) * (n_obs - 1) / (n_obs - k)

# Stock & Yogo (2005), Table 5.2
CRITICAL_VALUES_10PCT = {1: 16.38, 2: 19.93, ...}

# Symmetric Kaczmarz (Correia 2016, §3.2)
# Convergence tolerance 1e-8: matches reghdfe default

# Matches Stata reghdfe, vce(cluster)
dfc = n_groups / (n_groups - 1) * n_obs / (n_obs - dof_fe - k)
```

**In docstrings** -- author-year reference with brief method description, using Google-style format (matching pdoc):

```python
def ppml(formula, data, vcov="HC1", ...):
    """Poisson Pseudo-Maximum Likelihood (PPML) regression.

    Estimates E[y|x] = exp(x'beta) via iteratively reweighted least squares.

    Reference: Santos Silva and Tenreyro (2006), "The Log of Gravity",
    Review of Economics and Statistics.

    Args:
        ...
    """
```

This docstring from `_ppml.py:46-73` is the gold standard -- it has summary, extended description, full journal citation, complete Args with types/defaults, and Returns block.

## Conflicting Sources

When two references disagree on a formula:

1. **Cite both sources**
2. **State which is implemented and why**

**Worked example:** LIML sigma-squared uses `1/(n-k)` (textbook, finite-sample correction) vs `e'e/n` (Stata `ivregress`, asymptotic). We implement the Stata convention for parity, since matching Stata output is a core project goal:

```python
# LIML sigma^2: e'e/n (asymptotic, matches Stata ivregress)
# Textbook alternative: e'e/(n-k) (finite-sample correction)
sigma2 = float(resid @ resid) / n
```

## Numerical Engineering

Document stability choices with inline code comments explaining the rationale. These don't need textbook citations but do need explanation.

### Categories That Warrant Documentation

**Eigenvalue clamping** -- preventing negative values from floating-point roundoff:

```python
# Clamp to non-negative: generalized eigenvalue problem can produce
# small negative values for PSD matrices due to roundoff
kappa = max(eigenvalues.min(), 0.0)
```

**Probability clipping** -- preventing log(0) in log-likelihood:

```python
# Clip to (1e-15, 1-1e-15): prevents log(0) and log(1) overflow
# in log-likelihood computation
Phi = np.clip(Phi, 1e-15, 1 - 1e-15)
```

**Pseudoinverse fallback** -- handling singular matrices:

```python
# Fall back to pseudoinverse when X'X is singular
# (e.g., perfect collinearity after FE absorption)
try:
    beta = np.linalg.solve(XtX, Xty)
except np.linalg.LinAlgError:
    beta = np.linalg.pinv(XtX) @ Xty
```

**Convergence tolerances** -- what is being measured and why:

```python
# Convergence: max absolute change in demeaned values < tol
# Default 1e-8 matches Stata reghdfe. Tighter (1e-12) gives
# more Stata-exact results but ~2x more iterations on HDFE.
if np.max(np.abs(x_new - x_old)) < tol:
    break
```

**CG overflow detection** -- guarding against numerical divergence:

```python
# CG coefficient denominator too small: phantom zero-count groups
# from non-contiguous codes can make uv near-zero, causing overflow
if abs(uv) < 1e-30:
    break
```

**NaN-to-null boundary** -- IEEE NaN is not Polars null:

```python
# IEEE NaN passes through Polars drop_nulls() unchanged.
# Must convert NaN → null before dropping, or NaN propagates
# silently through all downstream computation.
```

See `docs/solutions/runtime-errors/fe-singleton-contiguity-and-edge-case-guards.md` for the full causal chain of how non-contiguous FE codes led to CG overflow and silent NaN corruption.

## Validation Expectations

### Parity Testing

New estimators should have Stata or R parity tests where feasible. Tolerance varies by estimator class:

| Context                | Default  | Rationale                              | Adjustable? |
|------------------------|----------|----------------------------------------|-------------|
| Demeaning convergence  | 1e-8     | Matches reghdfe default                | Yes (`tol=`) |
| OLS coefficients, iid SEs | 1e-6 | Direct computation, no iteration       | Per-test    |
| Stata parity (reghdfe) | 2e-5     | Demeaning algorithm differences        | Per-test    |
| LIML parity            | 2e-3     | Near-singular eigenvalue sensitivity   | Per-test    |
| Panel parity (RE, AB)  | 5e-2     | Iterative GLS, bootstrap-based tests   | Per-test    |
| Eigenvalue clamping    | max(0, λ)| Prevent negative variance from roundoff| No          |
| Prob clipping (probit) | 1e-15    | Prevent log(0) in log-likelihood       | No          |

**Document convergence metrics, not just thresholds.** Say "We check `max(abs(x_new - x_old)) < tol`" rather than just "tol=1e-8".

### Deterministic Testing

- Use `np.random.default_rng(seed)`, never bare `np.random.randn()`
- For bootstrap tests: either fix the seed or test weak properties that hold with overwhelming probability
- Non-deterministic test failures are bugs, not flakiness

## Key Formulas Requiring Citations

Author-year references for the most numerically consequential formulas. Exact equation lookups are a separate task.

| Formula area                | Primary source                          |
|-----------------------------|-----------------------------------------|
| Degree-of-freedom corrections (clustered SEs) | Cameron, Gelbach & Miller (2011) |
| Multi-way FE demeaning      | Correia (2016)                          |
| LIML eigenvalue approach    | Anderson & Rubin (1949)                 |
| GMM VCV                     | Hansen (1982)                           |
| GRS F-statistic             | Gibbons, Ross & Shanken (1989); Kamstra & Shi (2021) |
| Probit/Logit MLE            | Cameron & Trivedi (2005) or Greene (2018) |
| Newey-West / Bartlett kernel| Newey & West (1987)                     |
| Driscoll-Kraay              | Driscoll & Kraay (1998)                 |
| Arellano-Bond GMM           | Arellano & Bond (1991)                  |
| System GMM                  | Blundell & Bond (1998)                  |
| Swamy-Arora GLS             | Swamy & Arora (1972)                    |
| PPML                        | Santos Silva & Tenreyro (2006)          |
| Quantile regression         | Koenker & Bassett (1978)                |
| Panel FE/FD                 | Wooldridge (2010)                       |
| Fama-MacBeth risk premia    | Fama & MacBeth (1973)                   |
| Shanken correction          | Shanken (1992); Cochrane (2005) Ch. 12  |

## Bibliography

Full citations for all referenced texts, alphabetically by first author.

- Anderson, T.W. and H. Rubin. "Estimation of the Parameters of a Single Equation in a Complete System of Stochastic Equations." *Annals of Mathematical Statistics*, 20(1), 46-63, 1949.
- Arellano, M. and S. Bond. "Some Tests of Specification for Panel Data: Monte Carlo Evidence and an Application to Employment Equations." *Review of Economic Studies*, 58(2), 277-297, 1991.
- Blundell, R. and S. Bond. "Initial Conditions and Moment Restrictions in Dynamic Panel Data Models." *Journal of Econometrics*, 87(1), 115-143, 1998.
- Cameron, A.C. and P.K. Trivedi. *Microeconometrics: Methods and Applications.* Cambridge University Press, 2005.
- Cameron, A.C., J.B. Gelbach, and D.L. Miller. "Robust Inference with Multiway Clustering." *Journal of Business & Economic Statistics*, 29(2), 238-249, 2011.
- Cochrane, J.H. *Asset Pricing*. Revised ed., Princeton University Press, 2005.
- Correia, S. "Linear Models with High-Dimensional Fixed Effects: An Efficient and Feasible Estimator." Working paper, 2016.
- Driscoll, J.C. and A.C. Kraay. "Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data." *Review of Economics and Statistics*, 80(4), 549-560, 1998.
- Fama, E.F. and J.D. MacBeth. "Risk, Return, and Equilibrium: Empirical Tests." *Journal of Political Economy*, 81(3), 607-636, 1973.
- Gibbons, M.R., S.A. Ross, and J. Shanken. "A Test of the Efficiency of a Given Portfolio." *Econometrica*, 57(5), 1121-1152, 1989.
- Greene, W.H. *Econometric Analysis.* 8th ed., Pearson, 2018.
- Hansen, L.P. "Large Sample Properties of Generalized Method of Moments Estimators." *Econometrica*, 50(4), 1029-1054, 1982.
- Koenker, R. and G. Bassett. "Regression Quantiles." *Econometrica*, 46(1), 33-50, 1978.
- Kamstra, M.J. and R. Shi. "A Note on the GRS Test." Working paper, 2021.
- Newey, W.K. and K.D. West. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703-708, 1987.
- Santos Silva, J.M.C. and S. Tenreyro. "The Log of Gravity." *Review of Economics and Statistics*, 88(4), 641-658, 2006.
- Shanken, J. "On the Estimation of Beta-Pricing Models." *Review of Financial Studies*, 5(1), 1-33, 1992.
- Stock, J.H. and M. Yogo. "Testing for Weak Instruments in Linear IV Regression." In *Identification and Inference for Econometric Models*, ed. D.W.K. Andrews and J.H. Stock, Cambridge University Press, 2005.
- Swamy, P.A.V.B. and S.S. Arora. "The Exact Finite Sample Properties of the Estimators of Coefficients in the Error Components Regression Models." *Econometrica*, 40(2), 261-275, 1972.
- Wooldridge, J.M. *Econometric Analysis of Cross Section and Panel Data.* 2nd ed., MIT Press, 2010.
