"""Probit and Logit (binary choice) models via MLE."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy import stats

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import _clustered_meat
from polars_reg._utils import ensure_polars, extract_arrays


def _probit_ll_score_hess(beta, X, y):
    """Probit log-likelihood, score, and Hessian."""
    xb = X @ beta
    Phi = stats.norm.cdf(xb)
    phi = stats.norm.pdf(xb)

    # Clip to avoid log(0)
    Phi = np.clip(Phi, 1e-15, 1 - 1e-15)

    ll = np.sum(y * np.log(Phi) + (1 - y) * np.log(1 - Phi))

    # Score: lambda_i * x_i where lambda = y*phi/Phi - (1-y)*phi/(1-Phi)
    lam = y * phi / Phi - (1 - y) * phi / (1 - Phi)
    score = X.T @ lam

    # Hessian: -X' diag(w) X where w = phi^2 / (Phi*(1-Phi)) + lambda*xb
    w = phi**2 / (Phi * (1 - Phi)) + lam * xb
    H = -X.T @ (X * w[:, None])

    return ll, score, H, Phi, lam


def _logit_ll_score_hess(beta, X, y):
    """Logit log-likelihood, score, and Hessian."""
    xb = X @ beta
    # Numerically stable sigmoid
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


def _newton_raphson(ll_func, beta0, X, y, max_iter=100, tol=1e-8):
    """Newton-Raphson optimization for MLE."""
    beta = beta0.copy()
    for i in range(max_iter):
        ll, score, H, prob, _ = ll_func(beta, X, y)
        try:
            step = np.linalg.solve(H, score)
        except np.linalg.LinAlgError:
            warnings.warn("Singular Hessian in Newton-Raphson; using pseudoinverse")
            step = np.linalg.lstsq(H, score, rcond=None)[0]
        beta = beta - step
        if np.max(np.abs(step)) < tol:
            break
    else:
        warnings.warn(f"Newton-Raphson did not converge after {max_iter} iterations")

    ll, score, H, prob, lam_or_resid = ll_func(beta, X, y)
    return beta, ll, H, prob, lam_or_resid, score


def _binary_model(
    model_type: str,
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Common implementation for probit and logit."""
    if isinstance(cluster, str):
        cluster = [cluster]
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
    p_bar = y.mean()
    p_bar = np.clip(p_bar, 1e-15, 1 - 1e-15)
    ll_null = np.sum(y * np.log(p_bar) + (1 - y) * np.log(1 - p_bar))

    # Pseudo R² (McFadden)
    pseudo_r2 = 1.0 - ll / ll_null

    # VCV
    H_inv = np.linalg.inv(-H)  # default: inverse of information matrix

    if cluster:
        cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
        if model_type == "Probit":
            # Individual scores: lambda_i * x_i
            scores = X * score_resid[:, None]
        else:
            # Individual scores: (y_i - Lambda_i) * x_i
            scores = X * score_resid[:, None]

        if len(cluster_arrays_list) == 1:
            meat = _clustered_meat(X, score_resid, cluster_arrays_list[0])
            G = len(np.unique(cluster_arrays_list[0]))
            dfc = (G / (G - 1)) * ((n - 1) / (n - k))
            V = dfc * H_inv @ meat @ H_inv
        else:
            # Use sandwich with H_inv instead of (X'X)^{-1}
            # For MLE: V = H^{-1} M H^{-1} with clustered M
            V = _mle_multiway_clustered(X, score_resid, cluster_arrays_list, H_inv, n, k)
        vcov_type_out = "cluster"
        n_clusters = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters.values()) - 1
    elif vcov in ("HC1", "robust"):
        # Robust (sandwich): H^{-1} (sum s_i s_i') H^{-1}
        scores = X * score_resid[:, None]
        meat = scores.T @ scores
        dfc = n / (n - k)
        V = dfc * H_inv @ meat @ H_inv
        vcov_type_out = "robust"
        n_clusters = None
        df_r = n - k
    else:
        # Default: information matrix (Hessian-based)
        V = H_inv
        vcov_type_out = "iid"
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
    return result


def _mle_multiway_clustered(X, score_resid, cluster_list, H_inv, n, k):
    """Multi-way clustered VCV for MLE models (CGM inclusion-exclusion)."""
    from itertools import combinations

    from polars_reg._se import _interaction_codes

    D = len(cluster_list)
    V = np.zeros((k, k))
    dims = list(range(D))

    for size in range(1, D + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(dims, size):
            subset_arrays = [cluster_list[d] for d in subset]
            interaction = _interaction_codes(*subset_arrays)
            meat = _clustered_meat(X, score_resid, interaction)
            G = len(np.unique(interaction))
            dfc = (G / (G - 1)) * ((n - 1) / (n - k))
            V += sign * dfc * H_inv @ meat @ H_inv

    return V


def probit(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Probit regression (binary choice, normal link) via MLE.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        vcov: "iid" (information matrix) or "HC1"/"robust" (sandwich)
        cluster: Column name(s) for clustered SEs.
    """
    return _binary_model("Probit", formula, data, vcov=vcov, cluster=cluster)


def logit(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Logit regression (binary choice, logistic link) via MLE.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        vcov: "iid" (information matrix) or "HC1"/"robust" (sandwich)
        cluster: Column name(s) for clustered SEs.
    """
    return _binary_model("Logit", formula, data, vcov=vcov, cluster=cluster)


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
            J = density * (np.eye(k) - (1 - 2 * lam) * np.outer(beta, x_bar))
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
