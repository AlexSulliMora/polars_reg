"""polars_reg: Econometric regression methods using Polars DataFrames."""

from polars_reg._ols import ols
from polars_reg._results import RegressionResult

try:
    from polars_reg._iv import iv2sls
except ImportError:
    pass

try:
    from polars_reg._gmm import gmm_iv, liml
except ImportError:
    pass

try:
    from polars_reg._panel import panel_fd, panel_fe, panel_re
except ImportError:
    pass

from polars_reg._arellano_bond import panel_ab, panel_sys_gmm
from polars_reg._binary import logit, marginal_effects, odds_ratios, probit
from polars_reg._diagnostics import (
    hausman_test,
    kleibergen_paap_from_result,
    kleibergen_paap_test,
    weak_instrument_test,
)
from polars_reg._groupby import GroupRegressionResult, groupby_reg
from polars_reg._plotting import avplot, coefplot
from polars_reg._ppml import ppml
from polars_reg._quantile import quantreg
from polars_reg._regtable import RegTable, regtable
from polars_reg.r_equiv import compare_r, to_r
from polars_reg.stata import compare_stata, to_stata

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
    "to_stata",
    "compare_stata",
    "to_r",
    "compare_r",
    "hausman_test",
    "weak_instrument_test",
    "kleibergen_paap_test",
    "kleibergen_paap_from_result",
    "probit",
    "logit",
    "marginal_effects",
    "odds_ratios",
    "RegTable",
    "panel_ab",
    "panel_sys_gmm",
    "quantreg",
    "ppml",
    "coefplot",
    "avplot",
]
