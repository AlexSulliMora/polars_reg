"""Probit and Logit (binary choice) models via MLE.

Maximum likelihood estimation following Cameron & Trivedi (2005),
Microeconometrics: Methods and Applications, ch. 14.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import compute_vcov
from polars_reg._ssc import SSC, _default_ssc
from polars_reg._utils import ensure_polars, extract_arrays, validate_vcov


def _probit_ll_score_hess(
    beta: NDArray, X: NDArray, y: NDArray
) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
    """Probit log-likelihood, score, and Hessian."""
    xb = X @ beta
    Phi = stats.norm.cdf(xb)
    phi = stats.norm.pdf(xb)

    # Clip to (1e-15, 1-1e-15): prevents log(0) and log(1) in
    # log-likelihood y*log(Phi) + (1-y)*log(1-Phi)
    Phi = np.clip(Phi, 1e-15, 1 - 1e-15)

    ll = np.sum(y * np.log(Phi) + (1 - y) * np.log(1 - Phi))

    # Score: lambda_i * x_i where lambda = y*phi/Phi - (1-y)*phi/(1-Phi)
    lam = y * phi / Phi - (1 - y) * phi / (1 - Phi)
    score = X.T @ lam

    # Hessian: -X' diag(w) X where w = phi^2 / (Phi*(1-Phi)) + lambda*xb
    w = phi**2 / (Phi * (1 - Phi)) + lam * xb
    H = -X.T @ (X * w[:, None])

    return ll, score, H, Phi, lam


def _logit_ll_score_hess(
    beta: NDArray, X: NDArray, y: NDArray
) -> tuple[float, NDArray, NDArray, NDArray, NDArray]:
    """Logit log-likelihood, score, and Hessian."""
    xb = X @ beta
    # Numerically stable sigmoid, clipped to (1e-15, 1-1e-15) to
    # prevent log(0) in log-likelihood computation
    Lambda = 1.0 / (1.0 + np.exp(-xb))
    Lambda = np.clip(Lambda, 1e-15, 1 - 1e-15)

    ll = np.sum(y * np.log(Lambda) + (1 - y) * np.log(1 - Lambda))

    # Score: X'(y - Lambda)
    resid = y - Lambda
    score = X.T @ resid

    # Hessian: -X' diag(Lambda*(1-Lambda)) X
    w = Lambda * (1 - Lambda)
    H = -X.T @ (X * w[:, None])

    return ll, score, H, Lambda, resid


def _newton_raphson(
    ll_func: Callable[..., tuple[float, NDArray, NDArray, NDArray, NDArray]],
    beta0: NDArray,
    X: NDArray,
    y: NDArray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[NDArray, float, NDArray, NDArray, NDArray, NDArray]:
    """Newton-Raphson optimization for MLE.

    Convergence: max absolute Newton step < tol (default 1e-8).
    """
    beta = beta0.copy()
    for i in range(max_iter):
        ll, score, H, prob, _ = ll_func(beta, X, y)
        try:
            step = np.linalg.solve(H, score)
        except np.linalg.LinAlgError:
            warnings.warn("Singular Hessian in Newton-Raphson; using pseudoinverse", stacklevel=3)
            step = np.linalg.lstsq(H, score, rcond=None)[0]
        beta = beta - step
        if np.max(np.abs(step)) < tol:
            break
    else:
        warnings.warn(f"Newton-Raphson did not converge after {max_iter} iterations", stacklevel=3)

    ll, score, H, prob, lam_or_resid = ll_func(beta, X, y)
    return beta, ll, H, prob, lam_or_resid, score


def _binary_model(
    model_type: str,
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    ssc: SSC | None = None,
) -> RegressionResult:
    """Common implementation for probit and logit."""
    if ssc is None:
        ssc = _default_ssc()
    if isinstance(cluster, str):
        cluster = [cluster]
    _binary_vcov = {"iid", "HC1"}
    validate_vcov(vcov, _binary_vcov, model_type)
    data = ensure_polars(data)

    spec = parse_formula(formula)
    if spec.fe:
        raise ValueError(f"{model_type} does not support absorbed fixed effects")
    arrays = extract_arrays(data, spec, cluster=cluster)

    X, y = arrays.X, arrays.y
    n, k = X.shape

    # Validate binary outcome
    unique_y = np.unique(y)
    if not np.all(np.isin(unique_y, [0, 1])):
        raise ValueError(f"{model_type} requires binary (0/1) dependent variable")

    # Choose link function
    if model_type == "Probit":
        ll_func = _probit_ll_score_hess
    else:
        ll_func = _logit_ll_score_hess

    # Starting values: zeros
    beta0 = np.zeros(k)
    beta, ll, H, prob, score_resid, score_vec = _newton_raphson(ll_func, beta0, X, y)

    # Null model log-likelihood (intercept only)
    # Clip mean to prevent log(0) if all y=0 or all y=1
    p_bar = y.mean()
    p_bar = np.clip(p_bar, 1e-15, 1 - 1e-15)
    ll_null = np.sum(y * np.log(p_bar) + (1 - y) * np.log(1 - p_bar))

    # Pseudo R² (McFadden)
    pseudo_r2 = 1.0 - ll / ll_null

    # VCV
    H_inv = np.linalg.inv(-H)  # default: inverse of information matrix

    if cluster:
        cl_list = [arrays.cluster_arrays[c] for c in cluster]
        V = compute_vcov(X, score_resid, vcov, ssc, cluster_arrays=cl_list, bread=H_inv)
        vcov_type_out = "cluster"
        n_clusters = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters.values()) - 1
    else:
        V = compute_vcov(X, score_resid, vcov, ssc, bread=H_inv)
        vcov_type_out = vcov
        n_clusters = None
        df_r = n - k

    # Residuals (deviance residuals)
    resid = y - prob

    result = RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=arrays.names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=pseudo_r2,
        r_squared_adj=pseudo_r2,  # no adjusted pseudo R² standard
        model_type=model_type,
        vcov_type=vcov_type_out,
        n_clusters=n_clusters,
    )
    result._X = X
    result._y = y
    result._prob = prob
    result._ll = ll
    result._ll_null = ll_null
    result.ssc = ssc
    return result


def probit(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    ssc: SSC | None = None,
) -> RegressionResult:
    """Probit regression (binary choice, normal link) via MLE.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        vcov: "iid" (information matrix) or "HC1" (sandwich)
        cluster: Column name(s) for clustered SEs.
        ssc: Small-sample correction configuration. Default: pyfixest conventions.
    """
    return _binary_model("Probit", formula, data, vcov=vcov, cluster=cluster, ssc=ssc)


def logit(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    ssc: SSC | None = None,
) -> RegressionResult:
    """Logit regression (binary choice, logistic link) via MLE.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        vcov: "iid" (information matrix) or "HC1" (sandwich)
        cluster: Column name(s) for clustered SEs.
        ssc: Small-sample correction configuration. Default: pyfixest conventions.
    """
    return _binary_model("Logit", formula, data, vcov=vcov, cluster=cluster, ssc=ssc)


def marginal_effects(
    result: RegressionResult,
    at: str = "mean",
) -> pl.DataFrame:
    """Compute marginal effects from a probit or logit result.

    Args:
        result: A probit or logit RegressionResult.
        at: "mean" for marginal effects at the mean, or "average" for
            average marginal effects (AME).

    Returns:
        Polars DataFrame with columns: name, dy_dx, se, z, p.
    """
    if result.model_type not in ("Probit", "Logit"):
        raise ValueError("marginal_effects requires a Probit or Logit result")

    X = result._X
    beta = result.coefficients
    names = result.names
    k = len(beta)

    if at == "mean":
        # Evaluate at mean of X
        x_bar = X.mean(axis=0)
        xb = x_bar @ beta
        if result.model_type == "Probit":
            density = stats.norm.pdf(xb)
        else:
            lam = 1.0 / (1.0 + np.exp(-xb))
            density = lam * (1 - lam)
        dy_dx = density * beta
        # Delta method SE: d(dy/dx)/dbeta * V * d(dy/dx)/dbeta'
        # For probit at mean: d/dbeta_j [phi(x'b) * beta_j]
        #   = phi(x'b) * I_j - phi(x'b) * x'b * beta * x_j (via chain rule)
        # Simplified: Jacobian J[j, l] = phi * (delta_jl - xb * beta_j * x_l)
        if result.model_type == "Probit":
            J = density * (np.eye(k) - xb * np.outer(beta, x_bar))
        else:
            J = density * (np.eye(k) + (1 - 2 * lam) * np.outer(beta, x_bar))
        V_me = J @ result.vcov @ J.T
        se = np.sqrt(np.diag(V_me))
    elif at == "average":
        # Average marginal effects
        xb = X @ beta
        if result.model_type == "Probit":
            densities = stats.norm.pdf(xb)
        else:
            lam = 1.0 / (1.0 + np.exp(-xb))
            densities = lam * (1 - lam)
        dy_dx = densities.mean() * beta
        # Approximate SE: average density * SE(beta)
        se = densities.mean() * result.se
    else:
        raise ValueError(f"at must be 'mean' or 'average', got '{at}'")

    z = dy_dx / se
    p = 2.0 * stats.norm.sf(np.abs(z))

    return pl.DataFrame(
        {
            "name": names,
            "dy_dx": dy_dx,
            "se": se,
            "z": z,
            "p": p.tolist(),
        }
    )


def odds_ratios(result: RegressionResult) -> pl.DataFrame:
    """Compute odds ratios from a logit result.

    OR = exp(beta). Delta-method SE: se(OR) = OR * se(beta).

    Args:
        result: A Logit RegressionResult.

    Returns:
        Polars DataFrame with columns: name, or, se, z, p, ci_lower, ci_upper.
    """
    if result.model_type != "Logit":
        raise ValueError("odds_ratios requires a Logit result")

    or_vals = np.exp(result.coefficients)
    se_or = or_vals * result.se  # delta method
    z = np.log(or_vals) / result.se  # same as beta / se(beta)
    p = 2.0 * stats.norm.sf(np.abs(z))

    # CI on log scale, then exponentiate
    ci = result.confint()
    ci_lower = np.exp(ci[:, 0])
    ci_upper = np.exp(ci[:, 1])

    return pl.DataFrame(
        {
            "name": result.names,
            "or": or_vals,
            "se": se_or,
            "z": z,
            "p": p.tolist(),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )
