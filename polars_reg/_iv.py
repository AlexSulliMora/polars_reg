from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import (
    _clustered_meat,
    _interaction_codes,
    vcov_multiway_clustered,
)
from polars_reg._utils import extract_arrays


def iv2sls(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Two-Stage Least Squares (2SLS) IV regression.

    Args:
        formula: Formula string with IV syntax, e.g.
            "y ~ x_exog || x_endog ~ z1 + z2"          (no FE)
            "y ~ x_exog | fe | x_endog ~ z1 + z2"      (with FE)
        data: Polars DataFrame or LazyFrame
        vcov: "iid", "HC0", or "HC1"
        cluster: Column name(s) for clustered SEs. Overrides vcov.
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)

    if not spec.endog or not spec.instruments:
        raise ValueError(
            "IV formula must specify endogenous variables and instruments. "
            "Use syntax: y ~ x_exog || x_endog ~ z1 + z2"
        )

    arrays = extract_arrays(data, spec, cluster=cluster)

    X_exog = arrays.X  # exogenous regressors (may include intercept)
    y = arrays.y
    X_endog = arrays.endog  # endogenous regressors
    Z_excl = arrays.instruments  # excluded instruments

    n = arrays.n_obs
    k_exog = X_exog.shape[1]
    k_endog = X_endog.shape[1]
    k = k_exog + k_endog

    # Full instrument matrix: Z = [X_exog, Z_excluded]
    Z = np.column_stack([X_exog, Z_excl])

    # --- Stage 1: Project endogenous variables onto instrument space ---
    # Pz = Z (Z'Z)^{-1} Z'
    ZtZ = Z.T @ Z
    ZtZ_inv = np.linalg.inv(ZtZ)
    # X_endog_hat = Pz @ X_endog = Z @ (Z'Z)^{-1} @ Z' @ X_endog
    ZtX_endog = Z.T @ X_endog
    X_endog_hat = Z @ (ZtZ_inv @ ZtX_endog)

    # --- First-stage F-statistic (partial F-test of excluded instruments) ---
    first_stage_f = _first_stage_f(X_exog, X_endog, Z_excl)

    # --- Stage 2: 2SLS with X_hat as instrument for X ---
    # X_hat = [X_exog, X_endog_hat]
    # X     = [X_exog, X_endog]
    X = np.column_stack([X_exog, X_endog])
    X_hat = np.column_stack([X_exog, X_endog_hat])

    # beta = (X_hat' X)^{-1} X_hat' y
    XhX = X_hat.T @ X
    Xhy = X_hat.T @ y
    beta = np.linalg.solve(XhX, Xhy)

    # Residuals from ORIGINAL X
    resid = y - X @ beta

    # R-squared
    ss_res = resid @ resid
    y_demean = y - y.mean()
    ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    # --- Variance-covariance ---
    # Bread: (X_hat' X)^{-1}
    XhX_inv = np.linalg.inv(XhX)

    if cluster:
        cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
        if len(cluster_arrays_list) == 1:
            V = _iv_vcov_clustered(X_hat, resid, cluster_arrays_list[0], XhX_inv)
        else:
            V = _iv_vcov_multiway(X_hat, resid, cluster_arrays_list, XhX_inv)
        vcov_type = "cluster"
        n_clusters_dict = {
            c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster
        }
        df_r = min(n_clusters_dict.values()) - 1
    elif vcov == "iid":
        V = _iv_vcov_iid(X_hat, X, resid, XhX_inv)
        vcov_type = "iid"
        n_clusters_dict = None
        df_r = n - k
    else:
        V = _iv_vcov_robust(X_hat, resid, XhX_inv, kind=vcov)
        vcov_type = vcov
        n_clusters_dict = None
        df_r = n - k

    # Coefficient names: exog names + endog names
    names = arrays.names + arrays.endog_names

    return RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="2SLS",
        vcov_type=vcov_type,
        n_clusters=n_clusters_dict,
        first_stage_f=first_stage_f,
    )


def _first_stage_f(
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z_excl: np.ndarray,
) -> float:
    """Partial F-test of excluded instruments in first-stage regression.

    For each endogenous variable, regresses it on [X_exog] (restricted)
    and [X_exog, Z_excl] (unrestricted). Returns the F-stat for the
    first endogenous variable (standard practice for single-endog case).
    """
    n = X_exog.shape[0]
    k_exog = X_exog.shape[1]
    q = Z_excl.shape[1]  # number of excluded instruments

    # For each endogenous variable, compute partial F
    # (report for the first one, which is the standard single-endog case)
    x_end = X_endog[:, 0] if X_endog.ndim == 2 else X_endog

    # Restricted model: x_endog ~ X_exog
    beta_r = np.linalg.lstsq(X_exog, x_end, rcond=None)[0]
    resid_r = x_end - X_exog @ beta_r
    ss_r = resid_r @ resid_r

    # Unrestricted model: x_endog ~ X_exog + Z_excl
    Z_full = np.column_stack([X_exog, Z_excl])
    beta_u = np.linalg.lstsq(Z_full, x_end, rcond=None)[0]
    resid_u = x_end - Z_full @ beta_u
    ss_u = resid_u @ resid_u

    # F = ((SS_r - SS_u) / q) / (SS_u / (n - k_exog - q))
    f_stat = ((ss_r - ss_u) / q) / (ss_u / (n - k_exog - q))
    return float(f_stat)


def _iv_vcov_iid(
    X_hat: np.ndarray,
    X: np.ndarray,
    resid: np.ndarray,
    XhX_inv: np.ndarray,
) -> np.ndarray:
    """Homoskedastic VCV for 2SLS: sigma^2 * (X_hat'X)^{-1}."""
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    return sigma2 * XhX_inv


def _iv_vcov_robust(
    X_hat: np.ndarray,
    resid: np.ndarray,
    XhX_inv: np.ndarray,
    kind: str = "HC1",
) -> np.ndarray:
    """Heteroskedasticity-robust VCV for 2SLS.

    Sandwich: (X_hat'X)^{-1} meat (X_hat'X)^{-1}
    where meat = X_hat' diag(e^2) X_hat, with HC1 scaling.
    """
    n, k = X_hat.shape
    e2 = resid ** 2

    meat = X_hat.T @ (X_hat * e2[:, None])

    if kind == "HC0":
        return XhX_inv @ meat @ XhX_inv
    elif kind == "HC1":
        return (n / (n - k)) * XhX_inv @ meat @ XhX_inv
    else:
        raise ValueError(f"Unsupported robust SE kind for 2SLS: {kind}")


def _iv_vcov_clustered(
    X_hat: np.ndarray,
    resid: np.ndarray,
    clusters: np.ndarray,
    XhX_inv: np.ndarray,
) -> np.ndarray:
    """One-way cluster-robust VCV for 2SLS.

    Uses score vector X_hat * resid and bread (X_hat'X)^{-1}.
    """
    n, k = X_hat.shape
    meat = _clustered_meat(X_hat, resid, clusters)
    G = len(np.unique(clusters))
    dfc = (G / (G - 1)) * ((n - 1) / (n - k))
    return dfc * XhX_inv @ meat @ XhX_inv


def _iv_vcov_multiway(
    X_hat: np.ndarray,
    resid: np.ndarray,
    cluster_list: list[np.ndarray],
    XhX_inv: np.ndarray,
) -> np.ndarray:
    """Multi-way clustered VCV for 2SLS via Cameron-Gelbach-Miller."""
    from itertools import combinations

    D = len(cluster_list)
    n, k = X_hat.shape
    V = np.zeros((k, k))
    dims = list(range(D))

    for size in range(1, D + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(dims, size):
            subset_arrays = [cluster_list[d] for d in subset]
            interaction = _interaction_codes(*subset_arrays)
            G = len(np.unique(interaction))
            meat = _clustered_meat(X_hat, resid, interaction)
            dfc = (G / (G - 1)) * ((n - 1) / (n - k))
            V += sign * dfc * XhX_inv @ meat @ XhX_inv

    return V
