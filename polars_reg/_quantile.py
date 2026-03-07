"""Quantile regression via iteratively reweighted least squares (IRLS)."""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._utils import ensure_polars, extract_arrays


def _check_function(u: np.ndarray, tau: float) -> float:
    """Quantile regression check function: rho_tau(u) = u*(tau - I(u<0))."""
    return float(np.sum(u * (tau - (u < 0).astype(float))))


def _irls_quantreg(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve quantile regression via IRLS (interior point approximation).

    Uses the smoothed IRLS approach: iteratively solve weighted least squares
    with weights w_i = 1/max(|u_i|, epsilon).
    """
    n, k = X.shape
    eps = 1e-6

    # Starting values: OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(max_iter):
        resid = y - X @ beta
        # Weights for IRLS
        abs_resid = np.maximum(np.abs(resid), eps)
        # Asymmetric weights for quantile tau
        w = np.where(resid >= 0, tau, 1 - tau) / abs_resid

        # Weighted least squares step
        Xw = X * np.sqrt(w)[:, None]
        yw = y * np.sqrt(w)
        beta_new = np.linalg.lstsq(Xw, yw, rcond=None)[0]

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta


def quantreg(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    tau: float | list[float] = 0.5,
    n_boot: int = 200,
    seed: int | None = None,
) -> RegressionResult | list[RegressionResult]:
    """Quantile regression.

    Estimates conditional quantiles of the response variable.
    Uses IRLS for point estimates and bootstrap for inference.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        tau: Quantile(s) to estimate. Float for single, list for multiple.
            Default 0.5 (median regression).
        n_boot: Number of bootstrap replications for SE estimation.
        seed: Random seed for bootstrap reproducibility.

    Returns:
        Single RegressionResult for scalar tau, list for multiple quantiles.
    """
    if isinstance(tau, (list, tuple)):
        return [quantreg(formula, data, t, n_boot=n_boot, seed=seed) for t in tau]

    if not 0 < tau < 1:
        raise ValueError(f"tau must be between 0 and 1, got {tau}")

    data = ensure_polars(data)
    spec = parse_formula(formula)
    if spec.fe:
        raise ValueError("Quantile regression does not support absorbed fixed effects")
    arrays = extract_arrays(data, spec)

    X, y = arrays.X, arrays.y
    n, k = X.shape

    # Point estimates via IRLS
    beta = _irls_quantreg(X, y, tau)
    resid = y - X @ beta

    # Bootstrap SEs (pairs bootstrap)
    rng = np.random.default_rng(seed)
    betas_boot = np.empty((n_boot, k))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        betas_boot[b] = _irls_quantreg(X[idx], y[idx], tau)

    V = np.cov(betas_boot.T, ddof=1)

    # Pseudo R² (Koenker-Machado, 1999)
    obj_full = _check_function(resid, tau)
    obj_null = _check_function(y - np.quantile(y, tau), tau)
    pseudo_r2 = 1.0 - obj_full / obj_null if obj_null > 0 else 0.0

    df_r = n - k

    result = RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=arrays.names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=pseudo_r2,
        r_squared_adj=pseudo_r2,
        model_type=f"Quantile({tau:.2f})",
        vcov_type="bootstrap",
    )
    result._X = X
    result._y = y
    result._tau = tau
    return result
