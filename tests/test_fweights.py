"""Tests for frequency weights (fweights) in OLS."""

import numpy as np
import polars as pl
import pytest

from polars_reg import ols


@pytest.fixture
def fweight_data():
    """Dataset with frequency weights column."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.standard_normal(n) * 0.5
    fw = rng.integers(1, 6, size=n)  # integer weights 1-5
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "fw": fw.astype(float)})


def test_fweights_basic(fweight_data):
    """fweights runs without error and reports correct effective N."""
    r = ols("y ~ x1 + x2", data=fweight_data, fweights="fw")
    expected_n = int(fweight_data["fw"].sum())
    assert r.n_obs == expected_n
    assert r.model_type == "OLS (fweight)"


def test_fweights_matches_expanded():
    """fweights should produce same coefficients as manually expanded data."""
    rng = np.random.default_rng(123)
    n = 50
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.3
    fw = rng.integers(1, 4, size=n)

    df = pl.DataFrame({"y": y, "x1": x1, "fw": fw.astype(float)})

    # Fit with fweights
    r_fw = ols("y ~ x1", data=df, fweights="fw")

    # Manually expand and fit OLS
    y_exp = np.repeat(y, fw)
    x1_exp = np.repeat(x1, fw)
    df_exp = pl.DataFrame({"y": y_exp, "x1": x1_exp})
    r_exp = ols("y ~ x1", data=df_exp)

    np.testing.assert_allclose(r_fw.coefficients, r_exp.coefficients, rtol=1e-6)
    assert r_fw.n_obs == r_exp.n_obs


def test_fweights_se_positive(fweight_data):
    """SEs are positive."""
    r = ols("y ~ x1 + x2", data=fweight_data, fweights="fw")
    assert np.all(r.se > 0)


def test_fweights_dof(fweight_data):
    """DoF uses effective N = sum(f), not number of rows."""
    r = ols("y ~ x1 + x2", data=fweight_data, fweights="fw")
    expected_n = int(fweight_data["fw"].sum())
    expected_dof = expected_n - 3  # 3 = x1, x2, _cons
    assert r.df_r == expected_dof


def test_fweights_uniform_equals_ols(fweight_data):
    """Uniform fweights=1 should match plain OLS (same N)."""
    df = fweight_data.with_columns(pl.lit(1.0).alias("ones"))
    r_fw = ols("y ~ x1 + x2", data=df, fweights="ones")
    r_ols = ols("y ~ x1 + x2", data=df)
    np.testing.assert_allclose(r_fw.coefficients, r_ols.coefficients, rtol=1e-10)
    assert r_fw.n_obs == r_ols.n_obs


def test_fweights_non_integer_error():
    """Non-integer fweights raise an error."""
    df = pl.DataFrame({"y": [1.0, 2.0], "x1": [1.0, 2.0], "fw": [1.5, 2.0]})
    with pytest.raises(ValueError, match="positive integers"):
        ols("y ~ x1", data=df, fweights="fw")


def test_fweights_zero_error():
    """Zero fweights raise an error."""
    df = pl.DataFrame({"y": [1.0, 2.0], "x1": [1.0, 2.0], "fw": [0.0, 2.0]})
    with pytest.raises(ValueError, match="positive integers"):
        ols("y ~ x1", data=df, fweights="fw")


def test_fweights_and_weights_error():
    """Cannot specify both weights and fweights."""
    df = pl.DataFrame({"y": [1.0, 2.0], "x1": [1.0, 2.0], "w": [1.0, 1.0], "fw": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Cannot specify both"):
        ols("y ~ x1", data=df, weights="w", fweights="fw")


def test_fweights_summary(fweight_data):
    """Summary works with fweights."""
    r = ols("y ~ x1 + x2", data=fweight_data, fweights="fw")
    s = r.summary()
    assert "OLS (fweight)" in s
    assert str(r.n_obs) in s
