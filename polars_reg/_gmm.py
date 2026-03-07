"""LIML and two-step efficient GMM-IV estimators."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import linalg, stats

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import vcov_clustered, vcov_iid, vcov_multiway_clustered, vcov_robust
from polars_reg._utils import extract_arrays


def liml(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Limited Information Maximum Likelihood IV estimator.

    Args:
        formula: IV formula, e.g. "y ~ x_exog || x_endog ~ z1 + z2"
        data: Polars DataFrame or LazyFrame.
        vcov: "iid", "HC0", "HC1", "HC2", or "HC3".
        cluster: Column name(s) for clustered SEs. Overrides vcov.
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster)

    y = arrays.y
    X_exog = arrays.X  # includes intercept if add_intercept
    endog = arrays.endog
    instruments = arrays.instruments

    if endog is None or instruments is None:
        raise ValueError("LIML requires endogenous variables and instruments in the formula.")

    n = arrays.n_obs
    n_endog = endog.shape[1]
    n_instr = instruments.shape[1]

    if n_instr < n_endog:
        raise ValueError(
            f"Model is underidentified: {n_instr} instruments for {n_endog} endogenous variables."
        )

    # Full instrument matrix: Z = [X_exog, Z_excluded]
    Z = np.column_stack([X_exog, instruments])
    # Full regressor matrix: X_full = [X_exog, X_endog]
    X_full = np.column_stack([X_exog, endog])
    k = X_full.shape[1]

    names = arrays.names + (arrays.endog_names or [])

    # Compute kappa via the partialed-out formulation to avoid n x n matrices.
    # 1. Partial out X_exog from y, endog, and excluded instruments.
    XeXe_inv = np.linalg.inv(X_exog.T @ X_exog)
    y_tilde = y - X_exog @ (XeXe_inv @ (X_exog.T @ y))
    endog_tilde = endog - X_exog @ (XeXe_inv @ (X_exog.T @ endog))
    Z1_tilde = instruments - X_exog @ (XeXe_inv @ (X_exog.T @ instruments))

    # 2. Form Y* = [y_tilde, endog_tilde] and project onto partialed instruments.
    Y_star = np.column_stack([y_tilde, endog_tilde])
    Z1tZ1_inv = np.linalg.inv(Z1_tilde.T @ Z1_tilde)
    # Pz1 Y* = Z1_tilde (Z1_tilde'Z1_tilde)^{-1} Z1_tilde' Y*
    Pz1_Ystar = Z1_tilde @ (Z1tZ1_inv @ (Z1_tilde.T @ Y_star))

    # 3. A = Y*'Y* (total partialed variation), B = Y*'(I - Pz1)Y* (residual).
    A = Y_star.T @ Y_star
    B = A - Y_star.T @ Pz1_Ystar

    # 4. Solve generalised eigenvalue problem A v = kappa B v.
    #    Use scipy.linalg.eig which handles near-singular B.
    eigvals_gen = linalg.eig(A, B, right=False)
    # Keep finite real eigenvalues >= 1 (all valid kappas satisfy kappa >= 1).
    finite_mask = np.isfinite(eigvals_gen) & np.isreal(eigvals_gen)
    real_eigs = eigvals_gen[finite_mask].real
    valid_eigs = real_eigs[real_eigs >= 1.0 - 1e-6]
    kappa = float(np.min(valid_eigs))

    # LIML weight matrix: W = I - kappa * Mz = (1 - kappa) * I + kappa * Pz
    # Compute X_w = W @ X_full without forming the n x n matrix W.
    # Pz X_full = Z (Z'Z)^{-1} Z' X_full
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    Pz_Xfull = Z @ (ZtZ_inv @ (Z.T @ X_full))
    X_w = (1.0 - kappa) * X_full + kappa * Pz_Xfull

    # Also compute W @ y for the estimator
    Pz_y = Z @ (ZtZ_inv @ (Z.T @ y))
    y_w = (1.0 - kappa) * y + kappa * Pz_y

    # beta = (X_w' X_full)^{-1} X_w' y
    XwX = X_w.T @ X_full
    Xwy = X_w.T @ y
    beta = np.linalg.solve(XwX, Xwy)

    # Residuals from original X
    resid = y - X_full @ beta

    # R-squared
    ss_res = resid @ resid
    y_demean = y - y.mean()
    ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    # Variance-covariance
    if cluster:
        cluster_arrays = [arrays.cluster_arrays[c] for c in cluster]
        if len(cluster_arrays) == 1:
            V = vcov_clustered(X_w, resid, cluster_arrays[0])
        else:
            V = vcov_multiway_clustered(X_w, resid, cluster_arrays)
        vcov_type = "cluster"
        n_clusters = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters.values()) - 1
    elif vcov == "iid":
        # IID SE: sigma^2 * (X_w' X_full)^{-1}
        XwX_inv = np.linalg.inv(XwX)
        sigma2 = resid @ resid / (n - k)
        V = sigma2 * XwX_inv
        vcov_type = "iid"
        n_clusters = None
        df_r = n - k
    else:
        V = vcov_robust(X_w, resid, kind=vcov)
        vcov_type = vcov
        n_clusters = None
        df_r = n - k

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
        model_type="LIML",
        vcov_type=vcov_type,
        n_clusters=n_clusters,
    )


def gmm_iv(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Two-step efficient GMM-IV estimator with Hansen J test.

    Args:
        formula: IV formula, e.g. "y ~ x_exog || x_endog ~ z1 + z2"
        data: Polars DataFrame or LazyFrame.
        vcov: "iid", "HC0", "HC1", "HC2", or "HC3".
        cluster: Column name(s) for clustered SEs. Overrides vcov.
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster)

    y = arrays.y
    X_exog = arrays.X
    endog = arrays.endog
    instruments = arrays.instruments

    if endog is None or instruments is None:
        raise ValueError("GMM-IV requires endogenous variables and instruments in the formula.")

    n = arrays.n_obs
    n_endog = endog.shape[1]
    n_instr = instruments.shape[1]

    if n_instr < n_endog:
        raise ValueError(
            f"Model is underidentified: {n_instr} instruments for {n_endog} endogenous variables."
        )

    # Full instrument matrix: Z = [X_exog, Z_excluded]
    Z = np.column_stack([X_exog, instruments])
    # Full regressor matrix: X_full = [X_exog, X_endog]
    X_full = np.column_stack([X_exog, endog])
    k = X_full.shape[1]
    q = Z.shape[1]  # total number of moment conditions

    names = arrays.names + (arrays.endog_names or [])

    # Step 1: 2SLS with W = (Z'Z)^{-1}
    ZtZ = Z.T @ Z
    ZtZ_inv = np.linalg.inv(ZtZ)
    XtZ = X_full.T @ Z
    Zty = Z.T @ y

    # beta_1 = (X'Z (Z'Z)^{-1} Z'X)^{-1} X'Z (Z'Z)^{-1} Z'y
    A1 = XtZ @ ZtZ_inv @ XtZ.T
    b1 = XtZ @ ZtZ_inv @ Zty
    beta_1 = np.linalg.solve(A1, b1)

    # Step 1 residuals
    e1 = y - X_full @ beta_1

    # Step 2: Optimal weight matrix from step-1 residuals
    # S = Z' diag(e1^2) Z / n
    S = (Z * (e1**2)[:, None]).T @ Z / n

    S_inv = np.linalg.inv(S)

    # beta_2 = (X'Z S^{-1} Z'X)^{-1} X'Z S^{-1} Z'y
    A2 = XtZ @ S_inv @ XtZ.T
    b2 = XtZ @ S_inv @ Zty
    beta_2 = np.linalg.solve(A2, b2)

    # Step 2 residuals
    resid = y - X_full @ beta_2

    # VCV: n * (X'Z S^{-1} Z'X)^{-1}
    # where S = (1/n) Z' diag(e^2) Z, so V = n * (X'Z S_inv Z'X)^{-1}
    # Recompute S with step-2 residuals for final VCV and J test
    S_final = (Z * (resid**2)[:, None]).T @ Z / n
    S_final_inv = np.linalg.inv(S_final)

    A_final = XtZ @ S_final_inv @ XtZ.T
    V = np.linalg.inv(A_final) * n

    # Override with cluster VCV if requested
    if cluster:
        cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
        # For GMM with clustering, use sandwich form with Z-projected X
        # bread = (X'Z S^{-1} Z'X)^{-1}
        bread = np.linalg.inv(A_final)
        # Compute meat using cluster-robust formulation
        # score_i = Z_i' * e_i, effective X = Z S^{-1} Z' X
        # Use (X'Z S^{-1}) as the effective left-projection
        XZ_Sinv = XtZ @ S_final_inv  # k x q
        Xe = (XZ_Sinv @ Z.T).T * resid[:, None]  # n x k score-like
        if len(cluster_arrays_list) == 1:
            from polars_reg._se import _clustered_meat
            clusters = cluster_arrays_list[0]
            G = len(np.unique(clusters))
            meat = _clustered_meat(np.eye(k), np.ones(k), clusters)
            # Recompute meat properly
            unique_clusters = np.unique(clusters)
            meat_mat = np.zeros((k, k))
            for g in unique_clusters:
                mask = clusters == g
                sg = Xe[mask].sum(axis=0)
                meat_mat += np.outer(sg, sg)
            dfc = (G / (G - 1)) * ((n - 1) / (n - k))
            V = dfc * bread @ meat_mat @ bread / (n * n)
        vcov_type = "cluster"
        n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters_dict.values()) - 1
    else:
        vcov_type = "robust"  # GMM naturally uses heteroskedasticity-robust VCV
        n_clusters_dict = None
        df_r = n - k

    # R-squared
    ss_res = resid @ resid
    y_demean = y - y.mean()
    ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    # Hansen J test (overidentification)
    # j = n * g_bar' S_final^{-1} g_bar where g_bar = Z'e / n
    df_j = q - k  # degrees of freedom = number of overidentifying restrictions
    if df_j > 0:
        g_bar = Z.T @ resid / n
        j_stat = float(n * g_bar @ S_final_inv @ g_bar)
        j_pvalue = float(1.0 - stats.chi2.cdf(j_stat, df=df_j))
    else:
        j_stat = None
        j_pvalue = None

    return RegressionResult(
        coefficients=beta_2,
        vcov=V,
        residuals=resid,
        names=names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="GMM",
        vcov_type=vcov_type,
        n_clusters=n_clusters_dict if cluster else None,
        j_stat=j_stat,
        j_pvalue=j_pvalue,
    )
