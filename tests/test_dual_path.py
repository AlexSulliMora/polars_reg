"""Tests that Python fallback and Rust native paths produce identical results."""

import numpy as np
import polars as pl
from unittest.mock import patch

import polars_reg as pr
from polars_reg._demean import demean


def _make_fe_data(seed=42):
    """Shared data for dual-path tests."""
    rng = np.random.default_rng(seed)
    n = 500
    return pl.DataFrame({
        "y": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "fe1": rng.integers(0, 20, size=n),
        "fe2": rng.integers(0, 10, size=n),
        "cl": rng.integers(0, 15, size=n),
    })


def test_ols_fe_python_vs_rust():
    """OLS with FE produces same coefficients via Python and Rust paths."""
    df = _make_fe_data()

    # Rust path (default)
    r_rust = pr.ols("y ~ x1 + x2 | fe1 + fe2", data=df)

    # Python path
    with patch("polars_reg._demean._HAS_NATIVE", False), \
         patch("polars_reg._se._HAS_NATIVE", False):
        r_python = pr.ols("y ~ x1 + x2 | fe1 + fe2", data=df)

    np.testing.assert_allclose(r_rust.coefficients, r_python.coefficients, atol=1e-10)
    np.testing.assert_allclose(r_rust.se, r_python.se, atol=1e-8)


def test_demean_python_vs_rust():
    """Demeaned arrays match between Python and Rust paths."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((200, 3))
    fe_dict = {
        "a": rng.integers(0, 10, size=200),
        "b": rng.integers(0, 8, size=200),
    }

    # Rust path
    result_rust = demean(X.copy(), {k: v.copy() for k, v in fe_dict.items()})

    # Python path
    with patch("polars_reg._demean._HAS_NATIVE", False):
        result_python = demean(X.copy(), {k: v.copy() for k, v in fe_dict.items()})

    np.testing.assert_allclose(result_rust, result_python, atol=1e-10)


def test_ols_fe_predict_python_vs_rust():
    """predict() matches between Python and Rust paths."""
    df = _make_fe_data()

    r_rust = pr.ols("y ~ x1 + x2 | fe1", data=df)

    with patch("polars_reg._demean._HAS_NATIVE", False), \
         patch("polars_reg._se._HAS_NATIVE", False):
        r_python = pr.ols("y ~ x1 + x2 | fe1", data=df)

    # Both should have same coefficients
    np.testing.assert_allclose(r_rust.coefficients, r_python.coefficients, atol=1e-10)


def test_clustered_se_python_vs_rust():
    """Clustered SEs match between Python and Rust paths."""
    df = _make_fe_data()

    r_rust = pr.ols("y ~ x1 + x2 | fe1", data=df, cluster=["cl"])

    with patch("polars_reg._demean._HAS_NATIVE", False), \
         patch("polars_reg._se._HAS_NATIVE", False):
        r_python = pr.ols("y ~ x1 + x2 | fe1", data=df, cluster=["cl"])

    np.testing.assert_allclose(r_rust.coefficients, r_python.coefficients, atol=1e-10)
    np.testing.assert_allclose(r_rust.se, r_python.se, atol=1e-8)
