"""Diagnostic tests for regression models."""

from __future__ import annotations

import numpy as np
from scipy import stats

from polars_reg._results import RegressionResult


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
