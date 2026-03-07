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

from polars_reg._regtable import regtable
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
    "regtable",
    "to_stata",
    "compare_stata",
]
