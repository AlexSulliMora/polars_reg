import numpy as np
import polars as pl
import pytest


@pytest.fixture
def simple_data() -> pl.DataFrame:
    """Simple dataset for basic OLS tests."""
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    e = rng.standard_normal(n) * 0.5
    y = 2.0 + 1.5 * x1 - 0.5 * x2 + e
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture
def panel_data() -> pl.DataFrame:
    """Panel dataset with firm/year FE for reghdfe-style tests."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 50, 20
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(n_years), n_firms)
    firm_fe = rng.standard_normal(n_firms)
    year_fe = rng.standard_normal(n_years)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    e = rng.standard_normal(n) * 0.5
    y = 1.0 * x1 - 2.0 * x2 + firm_fe[firm_id] + year_fe[year_id] + e
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )


@pytest.fixture
def iv_data() -> pl.DataFrame:
    """IV dataset with endogenous regressor and instruments."""
    rng = np.random.default_rng(42)
    n = 1000
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    x_exog = rng.standard_normal(n)
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + u
    return pl.DataFrame(
        {
            "y": y,
            "x_endog": x_endog,
            "x_exog": x_exog,
            "z1": z1,
            "z2": z2,
        }
    )
