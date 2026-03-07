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
    from polars_reg._panel import panel_fe
except ImportError:
    pass

__all__ = ["ols", "iv2sls", "liml", "gmm_iv", "panel_fe", "RegressionResult"]
