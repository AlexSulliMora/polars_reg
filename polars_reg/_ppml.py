"""Poisson Pseudo-Maximum Likelihood (PPML) estimator.

Implements the PPML estimator of Santos Silva and Tenreyro (2006),
the workhorse for gravity models in trade economics. Estimates
E[y|x] = exp(x'beta) via IRLS / Newton-Raphson, using sandwich VCV
since only the conditional mean is assumed correct (not the full
Poisson distribution).
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import _clustered_meat, _mle_multiway_clustered, _recode_to_contiguous
from polars_reg._ssc import SSC, _default_ssc
from polars_reg._utils import ensure_polars, extract_arrays, validate_vcov


def _ppml_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))."""
    dev = np.zeros_like(y)
    pos = y > 0
    dev[pos] = y[pos] * np.log(y[pos] / mu[pos]) - (y[pos] - mu[pos])
    dev[~pos] = mu[~pos]  # when y=0: 0*log(0/mu) - (0-mu) = mu
    return 2.0 * float(np.sum(dev))


def _ppml_null_deviance(y: np.ndarray) -> float:
    """Null model deviance (intercept only): mu_0 = mean(y)."""
    mu0 = np.full_like(y, y.mean())
    return _ppml_deviance(y, mu0)


def ppml(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "HC1",
    cluster: list[str] | str | None = None,
    ssc: SSC | None = None,
    max_iter: int = 250,
    tol: float = 1e-8,
) -> RegressionResult:
    """Poisson Pseudo-Maximum Likelihood (PPML) regression.

    Estimates E[y|x] = exp(x'beta) via iteratively reweighted least squares.
    The Poisson assumption is only used for the conditional mean; inference
    uses sandwich (robust) standard errors by default. This makes PPML
    consistent even under misspecification of higher moments.

    Reference: Santos Silva and Tenreyro (2006), "The Log of Gravity",
    Review of Economics and Statistics.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2". Fixed effects
            (absorbed FE) are not supported.
        data: Polars DataFrame or LazyFrame (pandas DataFrames are
            auto-converted).
        vcov: Variance-covariance type. "HC1" (default) for sandwich VCV
            with n/(n-k) small-sample correction.
        cluster: Column name(s) for cluster-robust standard errors.
            Overrides vcov when provided.
        max_iter: Maximum number of IRLS iterations (default 250).
        tol: Convergence tolerance on max absolute change in beta
            (default 1e-8).

    Returns:
        RegressionResult with model_type="PPML". The predict() method
        returns X @ beta (the linear predictor). Apply np.exp() to obtain
        the conditional mean on the response scale.
    """
    if ssc is None:
        ssc = _default_ssc()
    if isinstance(cluster, str):
        cluster = [cluster]
    _ppml_vcov = {"iid", "HC1"}
    validate_vcov(vcov, _ppml_vcov, "PPML")
    data = ensure_polars(data)

    spec = parse_formula(formula)
    if spec.fe:
        raise ValueError("PPML does not support absorbed fixed effects")
    arrays = extract_arrays(data, spec, cluster=cluster)

    X, y = arrays.X, arrays.y
    n, k = X.shape

    # Validate non-negative outcome
    if np.any(y < 0):
        raise ValueError("PPML requires a non-negative dependent variable")

    # Initialize with OLS on log(y), replacing y=0 with 0.5 to avoid log(0)
    y_safe = np.where(y > 0, y, 0.5)
    beta = np.linalg.lstsq(X, np.log(y_safe), rcond=None)[0]

    converged = False
    for iteration in range(max_iter):
        mu = np.exp(X @ beta)
        # Clip conditional mean to (1e-10, 1e10): prevents underflow/overflow
        # in the Hessian X'diag(mu)X and deviance computation
        mu = np.clip(mu, 1e-10, 1e10)

        # Newton-Raphson step (Santos Silva & Tenreyro 2006, §2)
        # Score: X'(y - mu), Hessian: -X'diag(mu)X
        score_resid = y - mu  # n-vector of (y_i - mu_i)
        score = X.T @ score_resid
        hessian = -X.T @ (X * mu[:, None])

        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Singular Hessian in PPML Newton-Raphson; using pseudoinverse",
                stacklevel=2,
            )
            step = np.linalg.lstsq(hessian, score, rcond=None)[0]

        beta = beta - step
        if np.max(np.abs(step)) < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"PPML did not converge after {max_iter} iterations",
            stacklevel=2,
        )

    # Final mu
    mu = np.exp(X @ beta)
    mu = np.clip(mu, 1e-10, 1e10)
    score_resid = y - mu

    # Separation detection: |beta| > 10 suggests quasi-complete separation
    # (coefficient diverging because a regressor perfectly predicts y=0)
    if np.any(np.abs(beta) > 10):
        large_idx = np.where(np.abs(beta) > 10)[0]
        large_names = [arrays.names[i] for i in large_idx]
        warnings.warn(
            f"Possible separation detected: large coefficient(s) on "
            f"{', '.join(large_names)} (|beta| > 10). Results may be unreliable.",
            stacklevel=2,
        )
    if np.any(mu > 1e10):
        warnings.warn(
            "Possible separation detected: fitted values exceed 1e10. Results may be unreliable.",
            stacklevel=2,
        )

    # Hessian at convergence (information matrix)
    H = -X.T @ (X * mu[:, None])
    H_inv = np.linalg.inv(-H)  # (X'WX)^{-1} where W = diag(mu)

    # Goodness of fit: Pseudo R-squared based on deviance
    deviance = _ppml_deviance(y, mu)
    null_deviance = _ppml_null_deviance(y)
    pseudo_r2 = 1.0 - deviance / null_deviance if null_deviance > 0 else 0.0

    # VCV computation
    if cluster:
        cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]

        if len(cluster_arrays_list) == 1:
            from polars_reg._ssc import _compute_k_eff

            codes, G = _recode_to_contiguous(cluster_arrays_list[0])
            meat = _clustered_meat(X, score_resid, codes, G)
            k_eff = _compute_k_eff(k, ssc.k_fixef, 0, 0)
            k_adj_factor = (n - 1) / (n - k_eff) if ssc.k_adj else 1.0
            G_adj_factor = G / (G - 1) if ssc.G_adj else 1.0
            dfc = k_adj_factor * G_adj_factor
            V = dfc * H_inv @ meat @ H_inv
        else:
            V = _mle_multiway_clustered(X, score_resid, cluster_arrays_list, H_inv, n, k, ssc=ssc)
        vcov_type_out = "cluster"
        n_clusters = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters.values()) - 1
    elif vcov == "HC1":
        # Robust sandwich: H^{-1} M H^{-1} with n/(n-k) scaling
        from polars_reg._ssc import _compute_k_eff

        meat = X.T @ (X * (score_resid**2)[:, None])
        k_eff = _compute_k_eff(k, ssc.k_fixef, 0, 0)
        dfc = n / (n - k_eff) if ssc.k_adj else 1.0
        V = dfc * H_inv @ meat @ H_inv
        vcov_type_out = "HC1"
        n_clusters = None
        df_r = n - k
    else:
        # Hessian-based (assumes Poisson variance)
        V = H_inv
        vcov_type_out = "iid"
        n_clusters = None
        df_r = n - k

    # Residuals on response scale
    residuals = y - mu

    result = RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=residuals,
        names=arrays.names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=pseudo_r2,
        r_squared_adj=pseudo_r2,
        model_type="PPML",
        vcov_type=vcov_type_out,
        n_clusters=n_clusters,
    )
    result._X = X
    result._y = y
    result._mu = mu
    result._deviance = deviance
    result._null_deviance = null_deviance
    result.ssc = ssc
    return result
