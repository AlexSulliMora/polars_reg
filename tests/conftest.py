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


@pytest.fixture
def iv_data_panel() -> pl.DataFrame:
    """IV dataset with panel structure (entity + time)."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 50, 20
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(n_years), n_firms)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    x_exog = rng.standard_normal(n)
    firm_fe = rng.standard_normal(n_firms)
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + firm_fe[firm_id] + u
    return pl.DataFrame(
        {
            "y": y,
            "x_endog": x_endog,
            "x_exog": x_exog,
            "z1": z1,
            "z2": z2,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )


@pytest.fixture
def messy_data():
    """DataFrame with nulls, extreme values, singletons, and mixed types."""
    rng = np.random.default_rng(777)
    n = 500
    fe1_values = rng.integers(0, 50, size=n)
    fe1_values[:3] = [997, 998, 999]  # Force singletons
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x_const": np.ones(n),
            "x_with_nan": np.where(rng.random(n) < 0.05, np.nan, rng.standard_normal(n)),
            "fe1": fe1_values,
            "fe2": rng.integers(0, 30, size=n),
            "cluster1": rng.integers(0, 20, size=n),
            "entity": np.repeat(np.arange(50), 10),
            "time": np.tile(np.arange(10), 50),
            "z1": rng.standard_normal(n),
            "z2": rng.standard_normal(n),
            "x_endog": rng.standard_normal(n),
            "binary_y": rng.integers(0, 2, size=n),
            "count_y": rng.poisson(3, size=n).astype(float),
        }
    )
    return df
