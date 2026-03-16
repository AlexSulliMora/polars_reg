"""polars_reg: Econometric regression methods using Polars DataFrames."""

from polars_reg._arellano_bond import panel_ab, panel_sys_gmm
from polars_reg._binary import logit, marginal_effects, odds_ratios, probit
from polars_reg._compare import compare
from polars_reg._diagnostics import (
    GRSTestResult,
    grs_test,
    grs_test_from_group,
    hausman_test,
    kleibergen_paap_from_result,
    kleibergen_paap_test,
    weak_instrument_test,
)
from polars_reg._gmm import gmm_iv, liml
from polars_reg._groupby import GroupRegressionResult, groupby_reg
from polars_reg._iv import iv2sls
from polars_reg._ols import ols
from polars_reg._panel import panel_fd, panel_fe, panel_re
from polars_reg._plotting import avplot, coefplot
from polars_reg._ppml import ppml
from polars_reg._quantile import quantreg
from polars_reg._regtable import regtable
from polars_reg._results import RegressionResult
from polars_reg.r_equiv import to_r
from polars_reg.stata import to_stata

__all__ = [
    "ols",
    "iv2sls",
    "liml",
    "gmm_iv",
    "panel_fe",
    "panel_re",
    "panel_fd",
    "RegressionResult",
    "GroupRegressionResult",
    "groupby_reg",
    "regtable",
    "compare",
    "to_stata",
    "to_r",
    "hausman_test",
    "weak_instrument_test",
    "kleibergen_paap_test",
    "kleibergen_paap_from_result",
    "grs_test",
    "grs_test_from_group",
    "GRSTestResult",
    "probit",
    "logit",
    "marginal_effects",
    "odds_ratios",
    "panel_ab",
    "panel_sys_gmm",
    "quantreg",
    "ppml",
    "coefplot",
    "avplot",
]
