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
