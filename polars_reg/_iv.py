from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._demean import absorbed_dof, demean, drop_singletons
from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import (
    _clustered_meat,
    _interaction_codes,
)
from polars_reg._utils import ensure_polars, extract_arrays


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
    data = ensure_polars(data)

    spec = parse_formula(formula)

    if not spec.endog or not spec.instruments:
        raise ValueError(
            "IV formula must specify endogenous variables and instruments. "
            "Use syntax: y ~ x_exog || x_endog ~ z1 + z2"
        )

    arrays = extract_arrays(data, spec, cluster=cluster)

    X_exog = arrays.X
    y = arrays.y
    X_endog = arrays.endog
    Z_excl = arrays.instruments
    fe_dict = arrays.fe_arrays
    has_fe = len(fe_dict) > 0

    if X_endog is None or Z_excl is None:
        raise ValueError("IV requires endogenous variables and instruments.")

    # Handle absorbed FE: demean y, X_exog, X_endog, Z_excl
    if has_fe:
        keep = drop_singletons(fe_dict)
        if not keep.all():
            y = y[keep]
            X_exog = X_exog[keep]
            X_endog = X_endog[keep]
            Z_excl = Z_excl[keep]
            fe_dict = {k: v[keep] for k, v in fe_dict.items()}
            if cluster:
                arrays.cluster_arrays = {k: v[keep] for k, v in arrays.cluster_arrays.items()}

        # Remove intercept (absorbed by FE)
        if spec.add_intercept and arrays.names[-1] == "_cons":
            X_exog = X_exog[:, :-1]
            arrays.names = arrays.names[:-1]

        # Demean all variables
        all_vars = np.column_stack([y.reshape(-1, 1), X_exog, X_endog, Z_excl])
        demeaned = demean(all_vars, fe_dict)
        col = 0
        y = demeaned[:, col]
        col += 1
        X_exog = demeaned[:, col : col + X_exog.shape[1]]
        col += X_exog.shape[1]
        X_endog = demeaned[:, col : col + X_endog.shape[1]]
        col += X_endog.shape[1]
        Z_excl = demeaned[:, col:]

        df_abs = absorbed_dof(fe_dict)
        fe_absorbed = list(fe_dict.keys())
    else:
        df_abs = 0
        fe_absorbed = None

    n = len(y)
    k_exog = X_exog.shape[1]
    k_endog = X_endog.shape[1]
    k = k_exog + k_endog

    # Full instrument matrix: Z = [X_exog, Z_excluded]
    Z = np.column_stack([X_exog, Z_excl])

    # --- Stage 1: Project endogenous variables onto instrument space ---
    ZtZ = Z.T @ Z
    ZtZ_inv = np.linalg.inv(ZtZ)
    ZtX_endog = Z.T @ X_endog
    X_endog_hat = Z @ (ZtZ_inv @ ZtX_endog)

    # --- First-stage F-statistic (partial F-test of excluded instruments) ---
    first_stage_f = _first_stage_f(X_exog, X_endog, Z_excl)

    # --- Stage 2: 2SLS ---
    X = np.column_stack([X_exog, X_endog])
    X_hat = np.column_stack([X_exog, X_endog_hat])

    XhX = X_hat.T @ X
    Xhy = X_hat.T @ y
    beta = np.linalg.solve(XhX, Xhy)

    resid = y - X @ beta

    # R-squared (within-R² when FE are absorbed)
    ss_res = resid @ resid
    y_demean = y - y.mean()
    ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - df_abs)

    # --- Variance-covariance ---
    XhX_inv = np.linalg.inv(XhX)

    if cluster:
        cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
        if len(cluster_arrays_list) == 1:
            V = _iv_vcov_clustered(X_hat, resid, cluster_arrays_list[0], XhX_inv)
        else:
            V = _iv_vcov_multiway(X_hat, resid, cluster_arrays_list, XhX_inv)
        vcov_type = "cluster"
        n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters_dict.values()) - 1
    elif vcov == "iid":
        V = _iv_vcov_iid(X_hat, X, resid, XhX_inv, df_abs=df_abs)
        vcov_type = "iid"
        n_clusters_dict = None
        df_r = n - k - df_abs
    else:
        V = _iv_vcov_robust(X_hat, resid, XhX_inv, kind=vcov)
        vcov_type = vcov
        n_clusters_dict = None
        df_r = n - k - df_abs

    names = arrays.names + (arrays.endog_names or [])

    result = RegressionResult(
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
        fe_absorbed=fe_absorbed,
        df_absorbed=df_abs,
    )
    # Stash first-stage arrays for Kleibergen-Paap test
    result._X = X
    result._y = y
    result._iv_X_exog = X_exog
    result._iv_X_endog = X_endog
    result._iv_Z_excl = Z_excl
    result._iv_cluster_arrays = (
        [arrays.cluster_arrays[c] for c in cluster] if cluster else None
    )
    return result


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
    df_abs: int = 0,
) -> np.ndarray:
    """Homoskedastic VCV for 2SLS: sigma^2 * (X_hat'X)^{-1}."""
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k - df_abs)
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
    e2 = resid**2

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
