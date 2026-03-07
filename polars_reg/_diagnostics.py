"""Diagnostic tests for regression models."""

from __future__ import annotations

import numpy as np
from scipy import stats

from polars_reg._results import RegressionResult

# Stock & Yogo (2005) Table 5.2: Critical values for Cragg-Donald Wald F test
# k2=1 (single endogenous regressor), maximal IV relative bias
# Key: n_excluded_instruments -> {max_bias_pct: F_critical}
_STOCK_YOGO_K2_1 = {
    2: {5: 19.93, 10: 11.59, 20: 7.54, 30: 5.96},
    3: {5: 13.91, 10: 9.08, 20: 6.46, 30: 5.39},
    4: {5: 16.85, 10: 10.27, 20: 6.71, 30: 5.34},
    5: {5: 18.37, 10: 10.83, 20: 6.77, 30: 5.25},
    6: {5: 19.28, 10: 11.12, 20: 6.76, 30: 5.15},
    7: {5: 19.86, 10: 11.29, 20: 6.73, 30: 5.07},
    8: {5: 20.25, 10: 11.39, 20: 6.69, 30: 4.99},
    10: {5: 20.74, 10: 11.49, 20: 6.60, 30: 4.86},
    15: {5: 21.23, 10: 11.56, 20: 6.42, 30: 4.62},
    20: {5: 21.40, 10: 11.55, 20: 6.28, 30: 4.45},
    30: {5: 21.42, 10: 11.48, 20: 6.07, 30: 4.20},
}


def hausman_test(
    fe_result: RegressionResult,
    re_result: RegressionResult,
) -> dict:
    """Hausman specification test: FE vs RE.

    Tests H0: RE is consistent and efficient (no correlation between
    entity effects and regressors) against H1: FE is consistent but
    RE is not.

    Args:
        fe_result: Result from panel_fe()
        re_result: Result from panel_re()

    Returns:
        dict with 'statistic' (chi2), 'pvalue', 'df'
    """
    # Find common coefficient names (exclude _cons — RE has it, FE doesn't)
    fe_map = {n: i for i, n in enumerate(fe_result.names) if n != "_cons"}
    re_map = {n: i for i, n in enumerate(re_result.names) if n != "_cons"}
    common = [n for n in fe_map if n in re_map]

    if not common:
        raise ValueError("No common coefficients between FE and RE results")

    fe_idx = [fe_map[n] for n in common]
    re_idx = [re_map[n] for n in common]

    b_fe = fe_result.coefficients[fe_idx]
    b_re = re_result.coefficients[re_idx]
    V_fe = fe_result.vcov[np.ix_(fe_idx, fe_idx)]
    V_re = re_result.vcov[np.ix_(re_idx, re_idx)]

    diff = b_fe - b_re
    V_diff = V_fe - V_re

    # Ensure V_diff is positive semi-definite (numerical issues can make it not)
    eigvals = np.linalg.eigvalsh(V_diff)
    if np.any(eigvals < -1e-10):
        # Use pseudo-inverse for numerical stability
        chi2_stat = float(diff @ np.linalg.pinv(V_diff) @ diff)
    else:
        chi2_stat = float(diff @ np.linalg.solve(V_diff, diff))

    chi2_stat = max(0.0, chi2_stat)
    df = len(common)
    pvalue = float(1.0 - stats.chi2.cdf(chi2_stat, df))

    return {
        "statistic": chi2_stat,
        "pvalue": pvalue,
        "df": df,
        "coefficients_compared": common,
    }


def weak_instrument_test(
    iv_result: RegressionResult,
    n_instruments: int | None = None,
) -> dict:
    """Assess instrument strength for IV regression.

    Reports the first-stage F-statistic and applies the Staiger-Stock (1997)
    rule of thumb (F > 10). When the number of excluded instruments is known,
    also reports Stock-Yogo (2005) critical values for maximal relative bias.

    Args:
        iv_result: Result from iv2sls() or liml().
        n_instruments: Number of excluded instruments. Required for
            Stock-Yogo critical values.

    Returns:
        dict with 'f_stat', 'staiger_stock' (bool), 'stock_yogo' (dict or None)
    """
    if iv_result.first_stage_f is None:
        raise ValueError("No first-stage F-statistic available in the result")

    f = iv_result.first_stage_f

    result = {
        "f_stat": f,
        "staiger_stock": f > 10,
        "assessment": "strong" if f > 10 else "weak",
    }

    if n_instruments is not None and n_instruments in _STOCK_YOGO_K2_1:
        cv = _STOCK_YOGO_K2_1[n_instruments]
        result["stock_yogo"] = {
            "n_instruments": n_instruments,
            "critical_values": cv,
            "reject_5pct": f > cv[5],
            "reject_10pct": f > cv[10],
            "reject_20pct": f > cv[20],
            "reject_30pct": f > cv[30],
        }
    else:
        result["stock_yogo"] = None

    return result


def kleibergen_paap_test(
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z_excl: np.ndarray,
    resid_2sls: np.ndarray,
    vcov_type: str = "robust",
    cluster_arrays: list[np.ndarray] | None = None,
) -> dict:
    """Kleibergen-Paap (2006) rk Wald F-statistic for weak instruments.

    Generalizes the Cragg-Donald F-statistic to be robust to
    heteroskedasticity and clustering. For k2=1 (single endogenous),
    this is the robust first-stage Wald F. For k2>1, it is the minimum
    eigenvalue of the robust analog of the Cragg-Donald matrix.

    Can be compared against Stock-Yogo (2005) critical values.

    Args:
        X_exog: Exogenous regressors including intercept (n x k1).
        X_endog: Endogenous regressors (n x k2).
        Z_excl: Excluded instruments (n x l).
        resid_2sls: Second-stage residuals (n,). Not used in current
            implementation but kept for API compatibility.
        vcov_type: "robust" for HC-robust, "cluster" for cluster-robust.
        cluster_arrays: List of cluster code arrays (required when vcov_type="cluster").

    Returns:
        dict with 'rk_stat' (F-statistic), 'rk_raw' (chi2 before F-scaling).
    """
    n = X_exog.shape[0]
    k1 = X_exog.shape[1]
    k2 = X_endog.shape[1]
    n_excl = Z_excl.shape[1]

    # Partial out exogenous regressors from Z_excl and X_endog
    XeXe_inv = np.linalg.inv(X_exog.T @ X_exog)
    Mxe = lambda A: A - X_exog @ (XeXe_inv @ (X_exog.T @ A))  # noqa: E731

    Z_tilde = Mxe(Z_excl)
    Y_tilde = Mxe(X_endog)

    # First-stage: Y_tilde = Z_tilde @ Pi + V
    ZtZ = Z_tilde.T @ Z_tilde
    ZtZ_inv = np.linalg.inv(ZtZ)
    Pi_hat = ZtZ_inv @ (Z_tilde.T @ Y_tilde)  # n_excl x k2
    V_hat = Y_tilde - Z_tilde @ Pi_hat  # n x k2

    # Build robust VCV of vec(Pi_hat) using first-stage residuals
    if vcov_type == "cluster" and cluster_arrays is not None:
        from polars_reg._se import _interaction_codes

        if len(cluster_arrays) == 1:
            clusters = cluster_arrays[0]
        else:
            clusters = _interaction_codes(*cluster_arrays)
        _, codes = np.unique(clusters, return_inverse=True)
        G = codes.max() + 1
        dfc = G / (G - 1)

        Vcov_pi = np.zeros((k2, n_excl, n_excl))
        for m in range(k2):
            score_m = Z_tilde * V_hat[:, m : m + 1]
            Sm = np.zeros((G, n_excl))
            for j in range(n_excl):
                Sm[:, j] = np.bincount(codes, weights=score_m[:, j], minlength=G)
            meat_m = Sm.T @ Sm
            Vcov_pi[m] = dfc * ZtZ_inv @ meat_m @ ZtZ_inv

        Sigma_vv = np.zeros((k2, k2))
        for m1 in range(k2):
            for m2 in range(m1, k2):
                sv1 = np.zeros(G)
                sv2 = np.zeros(G)
                np.add.at(sv1, codes, V_hat[:, m1])
                np.add.at(sv2, codes, V_hat[:, m2])
                val = dfc * (sv1 @ sv2) / n
                Sigma_vv[m1, m2] = val
                Sigma_vv[m2, m1] = val
    else:
        # HC-robust
        k_fs = k1 + n_excl
        dfc = n / (n - k_fs)

        Vcov_pi = np.zeros((k2, n_excl, n_excl))
        for m in range(k2):
            score_m = Z_tilde * V_hat[:, m : m + 1]
            meat_m = score_m.T @ score_m
            Vcov_pi[m] = dfc * ZtZ_inv @ meat_m @ ZtZ_inv

        Sigma_vv = dfc * (V_hat.T @ V_hat) / n

    if k2 == 1:
        # Single endogenous: KP rk F = pi' V(pi)^{-1} pi / n_excl
        pi = Pi_hat[:, 0]
        V_pi = Vcov_pi[0]
        try:
            V_pi_inv = np.linalg.inv(V_pi)
        except np.linalg.LinAlgError:
            return {"rk_stat": None, "rk_raw": None}
        rk_chi2 = float(pi @ V_pi_inv @ pi)
        rk_f = rk_chi2 / n_excl
    else:
        # Multi-endogenous: minimum eigenvalue approach
        try:
            Sigma_vv_inv_half = _matrix_power(Sigma_vv, -0.5)
        except np.linalg.LinAlgError:
            return {"rk_stat": None, "rk_raw": None}

        V_avg = Vcov_pi.mean(axis=0)
        try:
            V_avg_inv = np.linalg.inv(V_avg)
        except np.linalg.LinAlgError:
            return {"rk_stat": None, "rk_raw": None}

        C = Pi_hat.T @ V_avg_inv @ Pi_hat
        M = Sigma_vv_inv_half @ C @ Sigma_vv_inv_half

        eigvals = np.linalg.eigvalsh(M)
        rk_chi2 = float(np.min(np.maximum(eigvals, 0.0)))
        df_num = n_excl - k2 + 1
        rk_f = rk_chi2 / max(df_num, 1)

    return {
        "rk_stat": rk_f,
        "rk_raw": rk_chi2,
    }


def kleibergen_paap_from_result(iv_result: RegressionResult) -> dict:
    """Compute Kleibergen-Paap rk F-stat directly from an iv2sls() result.

    The iv2sls() result must have been computed with vcov="HC1" or cluster=
    for the KP statistic to be meaningful (it's a robust test). With iid
    errors, it reduces to the Cragg-Donald statistic.

    Args:
        iv_result: Result from iv2sls().

    Returns:
        dict with 'rk_stat' and 'rk_raw'.
    """
    if not hasattr(iv_result, "_iv_X_exog"):
        raise ValueError(
            "Result does not contain first-stage arrays. "
            "Use iv2sls() (not liml or gmm_iv) to get these."
        )

    vcov_type = "cluster" if iv_result._iv_cluster_arrays else "robust"

    return kleibergen_paap_test(
        X_exog=iv_result._iv_X_exog,
        X_endog=iv_result._iv_X_endog,
        Z_excl=iv_result._iv_Z_excl,
        resid_2sls=iv_result.residuals,
        vcov_type=vcov_type,
        cluster_arrays=iv_result._iv_cluster_arrays,
    )


def _matrix_power(A: np.ndarray, p: float) -> np.ndarray:
    """Compute A^p via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    # Clamp small negative eigenvalues from numerical noise
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals**p) @ eigvecs.T
