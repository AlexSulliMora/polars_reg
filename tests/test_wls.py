"""Tests for Weighted Least Squares (WLS) via ols(..., weights=)."""

import numpy as np
import polars as pl
import pytest

from polars_reg import ols


@pytest.fixture
def wls_data():
    """Dataset with known heteroskedasticity for WLS testing."""
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # Heteroskedastic errors: variance proportional to 1/w
    w = rng.uniform(0.5, 5.0, n)
    e = rng.standard_normal(n) / np.sqrt(w)
    y = 2.0 + 1.5 * x1 - 0.5 * x2 + e
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "w": w})


@pytest.fixture
def wls_panel_data():
    """Panel data with weights for WLS + FE tests."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 30, 10
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(n_years), n_firms)
    firm_fe = rng.standard_normal(n_firms)
    x1 = rng.standard_normal(n)
    w = rng.uniform(0.5, 5.0, n)
    e = rng.standard_normal(n) / np.sqrt(w)
    y = 1.0 * x1 + firm_fe[firm_id] + e
    return pl.DataFrame({"y": y, "x1": x1, "w": w, "firm_id": firm_id, "year_id": year_id})


def test_wls_basic(wls_data):
    """WLS produces valid results and different coefficients from OLS."""
    r_ols = ols("y ~ x1 + x2", data=wls_data)
    r_wls = ols("y ~ x1 + x2", data=wls_data, weights="w")
    # Both should run without error
    assert r_wls.n_obs == r_ols.n_obs
    assert r_wls.model_type == "WLS"
    assert r_ols.model_type == "OLS"
    # Coefficients should differ (WLS is more efficient here)
    assert not np.allclose(r_wls.coefficients, r_ols.coefficients)


def test_wls_matches_manual(wls_data):
    """WLS matches manual pre-multiplication by sqrt(w)."""
    df = wls_data
    w = df["w"].to_numpy()
    sqw = np.sqrt(w * len(w) / w.sum())  # normalized
    y = df["y"].to_numpy()
    x1 = df["x1"].to_numpy()
    x2 = df["x2"].to_numpy()
    ones = np.ones(len(y))
    X = np.column_stack([x1, x2, ones])

    # Manual WLS
    Xw = X * sqw[:, None]
    yw = y * sqw
    beta_manual = np.linalg.lstsq(Xw, yw, rcond=None)[0]

    r_wls = ols("y ~ x1 + x2", data=df, weights="w")
    np.testing.assert_allclose(r_wls.coefficients, beta_manual, atol=1e-10)


def test_wls_uniform_weights_equals_ols(wls_data):
    """Uniform weights should give identical results to OLS."""
    df = wls_data.with_columns(pl.lit(1.0).alias("uniform_w"))
    r_ols = ols("y ~ x1 + x2", data=df)
    r_wls = ols("y ~ x1 + x2", data=df, weights="uniform_w")
    np.testing.assert_allclose(r_wls.coefficients, r_ols.coefficients, atol=1e-10)
    np.testing.assert_allclose(r_wls.se, r_ols.se, atol=1e-10)
    np.testing.assert_allclose(r_wls.r_squared, r_ols.r_squared, atol=1e-10)


def test_wls_negative_weights_error(wls_data):
    """Negative weights should raise an error."""
    df = wls_data.with_columns((pl.col("w") * -1).alias("neg_w"))
    with pytest.raises(ValueError, match="strictly positive"):
        ols("y ~ x1 + x2", data=df, weights="neg_w")


def test_wls_robust_se(wls_data):
    """WLS with robust SEs runs without error."""
    r = ols("y ~ x1 + x2", data=wls_data, weights="w", vcov="HC1")
    assert r.vcov_type == "HC1"
    assert r.se is not None
    assert np.all(r.se > 0)


def test_wls_clustered_se(wls_panel_data):
    """WLS with clustered SEs runs without error."""
    r = ols("y ~ x1", data=wls_panel_data, weights="w", cluster="firm_id")
    assert r.vcov_type == "cluster"
    assert r.se is not None
    assert np.all(r.se > 0)


def test_wls_with_fe(wls_panel_data):
    """WLS with absorbed fixed effects."""
    r = ols("y ~ x1 | firm_id", data=wls_panel_data, weights="w")
    assert r.fe_absorbed == ["firm_id"]
    assert r.model_type == "WLS"
    # Coefficient on x1 should be close to true value (1.0)
    assert abs(r.coefficients[0] - 1.0) < 0.2


def test_wls_with_fe_and_cluster(wls_panel_data):
    """WLS with FE absorption and clustered SEs."""
    r = ols("y ~ x1 | firm_id", data=wls_panel_data, weights="w", cluster="firm_id")
    assert r.fe_absorbed == ["firm_id"]
    assert r.vcov_type == "cluster"
    assert np.all(r.se > 0)


def test_wls_r_squared(wls_data):
    """WLS R² is between 0 and 1 and is a valid weighted R²."""
    r = ols("y ~ x1 + x2", data=wls_data, weights="w")
    assert 0 < r.r_squared < 1
    assert 0 < r.r_squared_adj < 1


def test_wls_summary(wls_data):
    """WLS summary displays correctly."""
    r = ols("y ~ x1 + x2", data=wls_data, weights="w")
    s = r.summary()
    assert "WLS" in s
    assert "x1" in s
    assert "x2" in s


def test_wls_coef_table(wls_data):
    """WLS coef_table returns valid Polars DataFrame."""
    r = ols("y ~ x1 + x2", data=wls_data, weights="w")
    ct = r.coef_table()
    assert ct.shape[0] == 3  # x1, x2, _cons
    assert "coef" in ct.columns
    assert "se" in ct.columns


def test_wls_scaling_invariance(wls_data):
    """Scaling weights by a constant shouldn't change results."""
    df = wls_data.with_columns((pl.col("w") * 100).alias("w_scaled"))
    r1 = ols("y ~ x1 + x2", data=df, weights="w")
    r2 = ols("y ~ x1 + x2", data=df, weights="w_scaled")
    np.testing.assert_allclose(r1.coefficients, r2.coefficients, atol=1e-10)
    np.testing.assert_allclose(r1.se, r2.se, atol=1e-10)


def test_wls_with_nan_in_weights():
    """WLS handles NaN in the weight column."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    y = 2 * x + rng.standard_normal(n) * 0.5
    w = (
        np.abs(
            rng.standard_normal(
                n,
            )
        )
        + 0.1
    )
    w[5] = float("nan")
    w[50] = float("nan")
    df = pl.DataFrame({"x": x, "y": y, "w": w})

    result = ols("y ~ x", data=df, weights="w")
    assert result.n_obs == n - 2  # 2 NaN weight rows dropped
    assert np.all(np.isfinite(result.coefficients))


def test_wls_with_inf_in_weights():
    """WLS handles inf in the weight column."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    y = 2 * x + rng.standard_normal(n) * 0.5
    w = np.abs(rng.standard_normal(n)) + 0.1
    w[0] = np.inf
    df = pl.DataFrame({"x": x, "y": y, "w": w})

    result = ols("y ~ x", data=df, weights="w")
    assert result.n_obs == n - 1
    assert np.all(np.isfinite(result.coefficients))
