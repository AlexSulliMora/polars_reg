"""Tests for quantile regression."""

import numpy as np
import polars as pl
import pytest

from polars_reg import quantreg


@pytest.fixture
def qreg_data():
    """Data for quantile regression tests."""
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # Heteroskedastic errors: variance increases with x1
    e = rng.standard_normal(n) * (1 + 0.5 * np.abs(x1))
    y = 2.0 + 1.5 * x1 - 0.5 * x2 + e
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_quantreg_median(qreg_data):
    """Median regression (tau=0.5) runs and produces valid output."""
    r = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.5, n_boot=99, seed=42)
    assert r.model_type == "Quantile(0.50)"
    assert r.n_obs == 1000
    assert np.all(r.se > 0)


def test_quantreg_median_close_to_ols(qreg_data):
    """Median regression coefficients should be close to OLS for symmetric errors."""
    from polars_reg import ols

    r_ols = ols("y ~ x1 + x2", data=qreg_data)
    r_med = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.5, n_boot=99, seed=42)
    # With heteroskedastic but roughly symmetric errors, median ~ OLS
    np.testing.assert_allclose(r_med.coefficients, r_ols.coefficients, atol=0.5)


def test_quantreg_coefficient_signs(qreg_data):
    """Coefficient signs should match the DGP."""
    r = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.5, n_boot=99, seed=42)
    idx_x1 = r.names.index("x1")
    idx_x2 = r.names.index("x2")
    assert r.coefficients[idx_x1] > 0  # true: 1.5
    assert r.coefficients[idx_x2] < 0  # true: -0.5


def test_quantreg_different_quantiles(qreg_data):
    """Different quantiles should give different intercepts."""
    r25 = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.25, n_boot=99, seed=42)
    r75 = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.75, n_boot=99, seed=42)
    idx_cons = r25.names.index("_cons")
    # Higher quantile should have higher intercept
    assert r75.coefficients[idx_cons] > r25.coefficients[idx_cons]


def test_quantreg_multiple_quantiles(qreg_data):
    """Passing a list of taus returns a list of results."""
    results = quantreg("y ~ x1 + x2", data=qreg_data, tau=[0.25, 0.5, 0.75], n_boot=99, seed=42)
    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0].model_type == "Quantile(0.25)"
    assert results[1].model_type == "Quantile(0.50)"
    assert results[2].model_type == "Quantile(0.75)"


def test_quantreg_pseudo_r2(qreg_data):
    """Pseudo R² should be between 0 and 1."""
    r = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.5, n_boot=99, seed=42)
    assert 0 < r.r_squared < 1


def test_quantreg_reproducible(qreg_data):
    """Same seed gives same SEs."""
    r1 = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.5, n_boot=99, seed=42)
    r2 = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.5, n_boot=99, seed=42)
    np.testing.assert_allclose(r1.se, r2.se)
    np.testing.assert_allclose(r1.coefficients, r2.coefficients)


def test_quantreg_invalid_tau(qreg_data):
    """Invalid tau raises error."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        quantreg("y ~ x1 + x2", data=qreg_data, tau=0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        quantreg("y ~ x1 + x2", data=qreg_data, tau=1.0)


def test_quantreg_no_fe(qreg_data):
    """FE raises error."""
    df = qreg_data.with_columns(pl.lit(1).alias("fe"))
    with pytest.raises(ValueError, match="does not support"):
        quantreg("y ~ x1 | fe", data=df, tau=0.5)


def test_quantreg_summary(qreg_data):
    """Summary displays correctly."""
    r = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.5, n_boot=99, seed=42)
    s = r.summary()
    assert "Quantile" in s
    assert "x1" in s


def test_quantreg_coef_table(qreg_data):
    """Coef table is valid."""
    r = quantreg("y ~ x1 + x2", data=qreg_data, tau=0.5, n_boot=99, seed=42)
    ct = r.coef_table()
    assert ct.shape[0] == 3
    assert "coef" in ct.columns
    assert "se" in ct.columns


# ── Additional robustness tests ───────────────────────────────────


def test_quantreg_nan_dropped():
    """NaN in x columns handled by dropping those rows."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 + 1.5 * x1 - 0.5 * x2 + rng.standard_normal(n) * 0.5
    x1[0] = np.nan
    x2[3] = np.nan
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    r = quantreg("y ~ x1 + x2", data=df, tau=0.5, n_boot=49, seed=42)
    assert r.n_obs == n - 2
    assert np.all(np.isfinite(r.coefficients))


def test_quantreg_null_dropped():
    """Polars nulls in x columns handled by dropping those rows."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 + 1.5 * x1 - 0.5 * x2 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    # Inject Polars nulls
    null_mask = pl.Series("m", [False] * n)
    null_mask[0] = True
    null_mask[3] = True
    df = df.with_columns(pl.when(null_mask).then(None).otherwise(pl.col("x1")).alias("x1"))
    assert df["x1"].null_count() == 2

    r = quantreg("y ~ x1 + x2", data=df, tau=0.5, n_boot=49, seed=42)
    assert r.n_obs == n - 2
    assert np.all(np.isfinite(r.coefficients))


def test_quantreg_inf_dropped():
    """Inf in x columns handled by dropping those rows."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 + 1.5 * x1 - 0.5 * x2 + rng.standard_normal(n) * 0.5
    # Inject inf
    x1[0] = np.inf
    x2[3] = -np.inf
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

    r = quantreg("y ~ x1 + x2", data=df, tau=0.5, n_boot=49, seed=42)
    assert r.n_obs == n - 2
    assert np.all(np.isfinite(r.coefficients))


def test_quantreg_lazyframe():
    """LazyFrame input works for quantreg."""
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1}).lazy()
    r = quantreg("y ~ x1", data=df, tau=0.5, n_boot=49, seed=42)
    assert r.n_obs == n
    assert np.all(np.isfinite(r.coefficients))


def test_quantreg_extreme_tau():
    """tau=0.01 and tau=0.99 produce stable results."""
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.standard_normal(n)
    y = 2.0 + 1.5 * x1 + rng.standard_normal(n)
    df = pl.DataFrame({"y": y, "x1": x1})
    r_low = quantreg("y ~ x1", data=df, tau=0.01, n_boot=49, seed=42)
    r_high = quantreg("y ~ x1", data=df, tau=0.99, n_boot=49, seed=42)
    assert np.all(np.isfinite(r_low.coefficients))
    assert np.all(np.isfinite(r_high.coefficients))
    assert np.all(r_low.se > 0)
    assert np.all(r_high.se > 0)
    # High-quantile intercept should be larger than low-quantile
    idx_cons = r_low.names.index("_cons")
    assert r_high.coefficients[idx_cons] > r_low.coefficients[idx_cons]
