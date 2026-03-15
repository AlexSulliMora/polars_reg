"""Diagnostic tests for regression models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats

from polars_reg._results import RegressionResult

if TYPE_CHECKING:
    from polars_reg._groupby import GroupRegressionResult

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

    # Ensure V_diff is positive semi-definite: V_FE - V_RE can have small
    # negative eigenvalues from roundoff. Threshold -1e-10 distinguishes
    # numerical noise from genuinely non-PSD differences.
    eigvals = np.linalg.eigvalsh(V_diff)
    if np.any(eigvals < -1e-10):
        # Use pseudo-inverse when V_diff is not PSD
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
    vcov_type: str = "HC",
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
        vcov_type: "HC" for HC-robust, "cluster" for cluster-robust.
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
            clusters, _ = _interaction_codes(*cluster_arrays)
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
        # Clamp to non-negative: M is theoretically PSD but roundoff
        # can produce small negative eigenvalues
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
    if iv_result._iv_X_exog is None:
        raise ValueError(
            "Result does not contain first-stage arrays. "
            "Use iv2sls() (not liml or gmm_iv) to get these."
        )

    vcov_type = "cluster" if iv_result._iv_cluster_arrays else "HC"

    return kleibergen_paap_test(
        X_exog=iv_result._iv_X_exog,
        X_endog=iv_result._iv_X_endog,
        Z_excl=iv_result._iv_Z_excl,
        vcov_type=vcov_type,
        cluster_arrays=iv_result._iv_cluster_arrays,
    )


@dataclass
class GRSTestResult:
    """Result of a GRS (Gibbons-Ross-Shanken 1989) F-test.

    Tests whether intercepts (alphas) from N time-series regressions are
    jointly zero — the standard test for asset pricing model specification.
    """

    # Core test
    statistic: float
    pvalue: float
    df: tuple[int, int]

    # Wald variant
    wald_statistic: float
    wald_pvalue: float

    # Dimensions
    n_assets: int
    n_periods: int
    n_factors: int

    # Alpha details
    alphas: NDArray
    alpha_names: list[str]
    alpha_se: NDArray

    # Sharpe ratio decomposition
    sharpe_sq_factors: float
    sharpe_sq_tangency: float

    # Matrices
    sigma: NDArray
    factor_means: NDArray
    factor_cov: NDArray

    def summary(self, precision: int = 4) -> str:
        """Formatted summary of GRS test results."""
        N, T, K = self.n_assets, self.n_periods, self.n_factors
        lines = [
            "GRS Test (Gibbons, Ross, Shanken 1989)",
            "\u2500" * 50,
            f"GRS F-statistic:  {self.statistic:{precision + 5}.{precision}f}"
            f"    F({N}, {self.df[1]})    p = {self.pvalue:.{precision}f}",
            f"Wald chi2:        {self.wald_statistic:{precision + 5}.{precision}f}"
            f"    chi2({N})      p = {self.wald_pvalue:.{precision}f}",
            "",
            f"N assets:   {N:>4}        N periods:  {T:>4}",
            f"N factors:  {K:>4}        df:         ({N}, {self.df[1]})",
            "",
            "Sharpe ratios (squared):",
            f"  Factor portfolio:    {self.sharpe_sq_factors:.{precision}f}",
            f"  Tangency portfolio:  {self.sharpe_sq_tangency:.{precision}f}",
            "",
            f"Alpha range: [{self.alphas.min():.{precision}f}, {self.alphas.max():.{precision}f}]",
        ]
        return "\n".join(lines)

    def critical_value(self, alpha: float = 0.05) -> float:
        """F critical value at the given significance level.

        Args:
            alpha: Significance level (default 0.05).

        Returns:
            Critical value from F(N, T-N-K) distribution.
        """
        return float(stats.f.ppf(1.0 - alpha, self.df[0], self.df[1]))

    def critical_values(self) -> dict[str, float]:
        """F critical values at standard significance levels.

        Returns:
            Dict with keys '10%', '5%', '1%' and their F critical values.
        """
        return {
            "10%": self.critical_value(0.10),
            "5%": self.critical_value(0.05),
            "1%": self.critical_value(0.01),
        }

    def alpha_table(self) -> pl.DataFrame:
        """Per-asset alpha table with t-statistics and p-values."""
        t_stats = self.alphas / self.alpha_se
        df_t = self.n_periods - self.n_factors - 1
        p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df_t))
        return pl.DataFrame(
            {
                "asset": self.alpha_names,
                "alpha": self.alphas,
                "se": self.alpha_se,
                "t": t_stats,
                "p": p_vals,
            }
        )

    def __repr__(self) -> str:
        return self.summary()


def _compute_grs(
    alphas: NDArray,
    residuals: NDArray,
    factor_data: NDArray | None,
    alpha_se: NDArray,
    alpha_names: list[str],
) -> GRSTestResult:
    """Core GRS computation (Kamstra & Shi 2021, eq. 7).

    Args:
        alphas: (N,) vector of estimated intercepts.
        residuals: (T, N) time-aligned residual matrix.
        factor_data: (T, K) factor data, or None if K=0.
        alpha_se: (N,) per-regression SEs of the intercept.
        alpha_names: Asset/portfolio labels.
    """
    T, N = residuals.shape
    K = 0 if factor_data is None else factor_data.shape[1]

    if T <= N + K:
        raise ValueError(f"GRS test requires T > N + K; got T={T}, N={N}, K={K}")

    # Sigma_hat: unbiased estimator (1/(T-K-1))
    Sigma_hat = (1.0 / (T - K - 1)) * residuals.T @ residuals
    try:
        Sigma_inv = np.linalg.inv(Sigma_hat)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Residual covariance matrix is singular — "
            "check for redundant or perfectly correlated assets"
        )

    alpha_Sigma_inv_alpha = float(alphas @ Sigma_inv @ alphas)

    if K == 0:
        mu_Omega_inv_mu = 0.0
        factor_means = np.array([])
        factor_cov = np.array([]).reshape(0, 0)
    else:
        mu = factor_data.mean(axis=0)
        # Omega_tilde: MLE estimator (1/T) — NOT np.cov which uses 1/(T-1)
        centered = factor_data - mu
        Omega_tilde = (1.0 / T) * centered.T @ centered
        try:
            Omega_inv = np.linalg.inv(Omega_tilde)
        except np.linalg.LinAlgError:
            raise ValueError("Factor covariance matrix is singular — check for collinear factors")
        mu_Omega_inv_mu = float(mu @ Omega_inv @ mu)
        factor_means = mu
        factor_cov = Omega_tilde

    # GRS F-statistic
    grs = (T / N) * ((T - N - K) / (T - K - 1)) * alpha_Sigma_inv_alpha / (1.0 + mu_Omega_inv_mu)
    df2 = T - N - K
    pvalue = float(1.0 - stats.f.cdf(grs, N, df2))

    # Wald chi-squared variant
    wald = N * grs
    wald_p = float(1.0 - stats.chi2.cdf(wald, N))

    # Sharpe ratio decomposition
    sh_f = mu_Omega_inv_mu
    sh_t = alpha_Sigma_inv_alpha + sh_f

    return GRSTestResult(
        statistic=float(grs),
        pvalue=pvalue,
        df=(N, df2),
        wald_statistic=float(wald),
        wald_pvalue=wald_p,
        n_assets=N,
        n_periods=T,
        n_factors=K,
        alphas=alphas,
        alpha_names=alpha_names,
        alpha_se=alpha_se,
        sharpe_sq_factors=sh_f,
        sharpe_sq_tangency=sh_t,
        sigma=Sigma_hat,
        factor_means=factor_means,
        factor_cov=factor_cov,
    )


def grs_test(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    *,
    assets: str,
    time: str,
) -> GRSTestResult:
    """GRS (Gibbons-Ross-Shanken 1989) F-test for asset pricing models.

    Tests whether intercepts (alphas) from N time-series regressions are
    jointly zero. Uses the correct multi-factor formula from Kamstra & Shi
    (2021) with asymmetric Sigma/Omega pairing for an exact F distribution.

    Args:
        formula: Formula string (e.g. "ret ~ mktrf + smb + hml").
            Must include an intercept (no ``-1``).
        data: Long-format panel with one row per (asset, time).
        assets: Column name identifying test assets/portfolios.
        time: Column name for the time dimension.

    Returns:
        GRSTestResult with F-statistic, p-value, Wald test, and diagnostics.
    """
    from polars_reg._formula import parse_formula
    from polars_reg._utils import ensure_polars

    data = ensure_polars(data)
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    spec = parse_formula(formula)
    if not spec.add_intercept:
        raise ValueError("GRS test requires an intercept in each regression")
    if spec.fe:
        raise ValueError("GRS test does not support fixed effects in the formula")
    if spec.endog:
        raise ValueError("GRS test does not support IV formulas")

    depvar = spec.depvar
    factor_names = spec.exog

    # Validate columns exist
    for col in [depvar, assets, time] + factor_names:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    # Sort by (assets, time) to guarantee alignment
    data = data.sort([assets, time])

    # Get unique assets
    asset_values = data[assets].unique(maintain_order=False).sort().to_list()
    len(asset_values)

    # Validate balanced panel and collect per-asset data
    alphas_list = []
    alpha_se_list = []
    residuals_list = []
    T = None

    for asset_val in asset_values:
        mask = data[assets] == asset_val
        asset_df = data.filter(mask)

        # Drop rows with any null in relevant columns
        cols = [depvar] + factor_names
        asset_df = asset_df.drop_nulls(subset=cols)

        if T is None:
            T = len(asset_df)
        elif len(asset_df) != T:
            raise ValueError(
                "Unbalanced panel: assets have different observation counts. "
                "GRS requires a balanced panel."
            )

        # Extract arrays and run OLS
        y = asset_df[depvar].to_numpy().astype(np.float64)
        K = len(factor_names)
        if K > 0:
            X = np.column_stack([asset_df[c].to_numpy().astype(np.float64) for c in factor_names])
            X = np.column_stack([X, np.ones(T)])
        else:
            X = np.ones((T, 1))

        # OLS: beta = (X'X)^{-1} X'y
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        alpha = beta[-1]  # intercept is last column
        # SE of intercept: sqrt(sigma^2 * (X'X)^{-1}[-1,-1])
        s2 = float(resid @ resid) / (T - K - 1)
        XtX_inv = np.linalg.inv(X.T @ X)
        se_alpha = float(np.sqrt(s2 * XtX_inv[-1, -1]))

        alphas_list.append(alpha)
        alpha_se_list.append(se_alpha)
        residuals_list.append(resid)

    alphas_arr = np.array(alphas_list)
    alpha_se_arr = np.array(alpha_se_list)
    residuals_mat = np.column_stack(residuals_list)  # (T, N)
    alpha_names = [str(v) for v in asset_values]

    # Factor data from first asset's rows (identical across assets)
    first_asset_df = data.filter(data[assets] == asset_values[0]).drop_nulls(
        subset=[depvar] + factor_names
    )
    if K > 0:
        factor_data = np.column_stack(
            [first_asset_df[c].to_numpy().astype(np.float64) for c in factor_names]
        )
    else:
        factor_data = None

    return _compute_grs(alphas_arr, residuals_mat, factor_data, alpha_se_arr, alpha_names)


def grs_test_from_group(
    group_result: "GroupRegressionResult",
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    *,
    assets: str,
    time: str,
) -> GRSTestResult:
    """GRS F-test from an existing GroupRegressionResult.

    Re-computes time-aligned residuals from the stored coefficients to
    guarantee cross-asset alignment.

    Args:
        group_result: Result from groupby_reg().
        formula: Formula string used for the regressions.
        data: Original data (needed for factor moments and residual alignment).
        assets: Column name identifying test assets/portfolios.
        time: Column name for the time dimension.

    Returns:
        GRSTestResult with F-statistic, p-value, Wald test, and diagnostics.
    """
    from polars_reg._formula import parse_formula
    from polars_reg._groupby import GroupRegressionResult
    from polars_reg._utils import ensure_polars

    if not isinstance(group_result, GroupRegressionResult):
        raise TypeError("group_result must be a GroupRegressionResult")

    if group_result.failed:
        failed_keys = list(group_result.failed.keys())
        raise ValueError(
            f"GRS test requires all group regressions to succeed; failed: {failed_keys}"
        )

    if len(group_result) == 0:
        raise ValueError("GroupRegressionResult has no successful results")

    data = ensure_polars(data)
    if isinstance(data, pl.LazyFrame):
        data = data.collect()

    spec = parse_formula(formula)
    if not spec.add_intercept:
        raise ValueError("GRS test requires an intercept in each regression")

    depvar = spec.depvar
    factor_names = spec.exog

    # Sort data by (assets, time) for alignment
    data = data.sort([assets, time])

    # Extract alphas and SEs from group results
    asset_keys = list(group_result.keys())
    len(asset_keys)
    alpha_names = [str(k) for k in asset_keys]

    alphas_list = []
    alpha_se_list = []

    for key in asset_keys:
        result = group_result[key]
        if "_cons" not in result.names:
            raise ValueError(
                f"Group '{key}' has no intercept ('_cons'). "
                "GRS test requires an intercept in each regression."
            )
        cons_idx = list(result.names).index("_cons")
        alphas_list.append(result.coefficients[cons_idx])
        alpha_se_list.append(result.se[cons_idx])

    alphas_arr = np.array(alphas_list)
    alpha_se_arr = np.array(alpha_se_list)

    # Re-compute aligned residuals from data using stored coefficients
    K = len(factor_names)
    residuals_list = []
    T = None

    for i, key in enumerate(asset_keys):
        mask = data[assets] == key
        asset_df = data.filter(mask)
        asset_df = asset_df.drop_nulls(subset=[depvar] + factor_names)

        if T is None:
            T = len(asset_df)
        elif len(asset_df) != T:
            raise ValueError(
                "Unbalanced panel: assets have different observation counts. "
                "GRS requires a balanced panel."
            )

        y = asset_df[depvar].to_numpy().astype(np.float64)
        result = group_result[key]

        # Reconstruct residuals: y - X @ beta_slope - alpha
        fitted = np.full(T, alphas_list[i])
        for name in factor_names:
            if name in result.names:
                coef_idx = list(result.names).index(name)
                fitted += result.coefficients[coef_idx] * asset_df[name].to_numpy().astype(
                    np.float64
                )

        resid = y - fitted
        residuals_list.append(resid)

    residuals_mat = np.column_stack(residuals_list)  # (T, N)

    # Factor data from first asset's rows
    first_asset_df = data.filter(data[assets] == asset_keys[0]).drop_nulls(
        subset=[depvar] + factor_names
    )
    if K > 0:
        factor_data = np.column_stack(
            [first_asset_df[c].to_numpy().astype(np.float64) for c in factor_names]
        )
    else:
        factor_data = None

    return _compute_grs(alphas_arr, residuals_mat, factor_data, alpha_se_arr, alpha_names)


def _matrix_power(A: np.ndarray, p: float) -> np.ndarray:
    """Compute A^p via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    # Clamp small negative eigenvalues from numerical noise
    eigvals = np.maximum(eigvals, 0.0)
    if p < 0 and np.any(eigvals < 1e-14):
        raise ValueError("Matrix is singular; cannot compute negative matrix power.")
    return eigvecs @ np.diag(eigvals**p) @ eigvecs.T
