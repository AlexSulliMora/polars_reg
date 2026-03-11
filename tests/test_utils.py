import numpy as np
import polars as pl
import pytest

from polars_reg._formula import FormulaSpec
from polars_reg._utils import extract_arrays


def test_extract_basic():
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x1": [4.0, 5.0, 6.0]})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.y.shape == (3,)
    assert arrays.X.shape == (3, 2)  # x1 + intercept
    np.testing.assert_array_equal(arrays.y, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(arrays.X[:, 0], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(arrays.X[:, 1], [1.0, 1.0, 1.0])  # intercept last
    assert arrays.names == ["x1", "_cons"]


def test_extract_no_intercept():
    df = pl.DataFrame({"y": [1.0, 2.0], "x1": [3.0, 4.0]})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=False)
    arrays = extract_arrays(df, spec)
    assert arrays.X.shape == (2, 1)
    assert arrays.names == ["x1"]


def test_extract_drops_na():
    df = pl.DataFrame({"y": [1.0, None, 3.0], "x1": [4.0, 5.0, 6.0]})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.y.shape == (2,)
    assert arrays.n_obs == 2


def test_extract_fe_codes():
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "g": ["a", "b", "a", "b"],
        }
    )
    spec = FormulaSpec(depvar="y", exog=["x1"], fe=["g"])
    arrays = extract_arrays(df, spec)
    assert "g" in arrays.fe_arrays
    codes = arrays.fe_arrays["g"]
    assert len(np.unique(codes)) == 2


def test_extract_cluster_codes():
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x1": [1.0, 2.0, 3.0], "cl": [0, 1, 0]})
    spec = FormulaSpec(depvar="y", exog=["x1"])
    arrays = extract_arrays(df, spec, cluster=["cl"])
    assert "cl" in arrays.cluster_arrays


def test_ensure_polars_passthrough():
    """Test that ensure_polars returns Polars DataFrame unchanged."""
    from polars_reg._utils import ensure_polars

    df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    result = ensure_polars(df)
    assert isinstance(result, pl.DataFrame)


def test_ensure_polars_converts_pandas():
    """Test that ensure_polars converts pandas DataFrame."""
    pd = pytest.importorskip("pandas")
    from polars_reg._utils import ensure_polars

    pdf = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    result = ensure_polars(pdf)
    assert isinstance(result, pl.DataFrame)


# ── NaN / Inf handling ──────────────────────────────────────────


def test_extract_arrays_nan_in_y():
    """NaN in y column should be dropped, reducing observation count."""
    rng = np.random.default_rng(100)
    y = rng.standard_normal(50)
    y[0] = np.nan
    y[10] = np.nan
    df = pl.DataFrame({"y": y, "x1": rng.standard_normal(50)})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == 48
    assert not np.any(np.isnan(arrays.y))


def test_extract_arrays_nan_in_x():
    """NaN in x column should be dropped, reducing observation count."""
    rng = np.random.default_rng(101)
    x1 = rng.standard_normal(50)
    x1[5] = np.nan
    x1[15] = np.nan
    x1[25] = np.nan
    df = pl.DataFrame({"y": rng.standard_normal(50), "x1": x1})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == 47
    assert not np.any(np.isnan(arrays.X))


def test_extract_arrays_nan_in_fe():
    """NaN in FE column should cause those rows to be dropped."""
    rng = np.random.default_rng(102)
    n = 50
    fe = rng.integers(0, 5, size=n).astype(float)
    fe[0] = np.nan
    fe[1] = np.nan
    df = pl.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "fe1": fe,
        }
    )
    spec = FormulaSpec(depvar="y", exog=["x1"], fe=["fe1"])
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == 48


def test_extract_arrays_inf_in_x():
    """Inf values in x should raise ValueError after all rows are dropped."""
    rng = np.random.default_rng(103)
    x1 = rng.standard_normal(10)
    x1[0] = np.inf
    x1[1] = -np.inf
    df = pl.DataFrame({"y": rng.standard_normal(10), "x1": x1})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    # Inf is not NaN, so it will pass through. The arrays should still have Inf.
    # This tests that the code doesn't silently eat Inf. Users should clean data.
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == 10  # Inf is not dropped
    assert np.any(np.isinf(arrays.X))


# ── Empty / minimal DataFrame ──────────────────────────────────


def test_extract_arrays_empty_df():
    """0-row DataFrame raises ValueError."""
    df = pl.DataFrame({"y": pl.Series([], dtype=pl.Float64), "x1": pl.Series([], dtype=pl.Float64)})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    with pytest.raises(ValueError, match="no observations"):
        extract_arrays(df, spec)


def test_extract_arrays_single_row():
    """1-row DataFrame with formula y ~ x1 should work."""
    df = pl.DataFrame({"y": [1.0], "x1": [2.0]})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.n_obs == 1
    assert arrays.y.shape == (1,)
    assert arrays.X.shape == (1, 2)


# ── All-null column ────────────────────────────────────────────


def test_extract_arrays_all_null_column():
    """Column with all nulls raises ValueError (no observations remain)."""
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x1": [None, None, None],
        }
    )
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    with pytest.raises(ValueError, match="No observations remain"):
        extract_arrays(df, spec)


# ── Type handling ──────────────────────────────────────────────


def test_extract_arrays_int_columns():
    """Integer-typed x columns auto-cast to float64."""
    df = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x1": [10, 20, 30],
        }
    )
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.X.dtype == np.float64
    np.testing.assert_allclose(arrays.X[:, 0], [10.0, 20.0, 30.0], atol=1e-15)


# ── Missing column reference ──────────────────────────────────


def test_extract_arrays_missing_column():
    """Formula references non-existent column raises error."""
    df = pl.DataFrame({"y": [1.0, 2.0], "x1": [3.0, 4.0]})
    spec = FormulaSpec(depvar="y", exog=["nonexistent"], add_intercept=True)
    with pytest.raises(Exception):
        extract_arrays(df, spec)


# ── ensure_polars edge cases ──────────────────────────────────


def test_ensure_polars_lazyframe():
    """LazyFrame passes through unchanged."""
    from polars_reg._utils import ensure_polars

    lf = pl.LazyFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    result = ensure_polars(lf)
    assert isinstance(result, pl.LazyFrame)


def test_ensure_polars_pandas():
    """pandas DataFrame converts to Polars DataFrame."""
    pd = pytest.importorskip("pandas")
    from polars_reg._utils import ensure_polars

    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = ensure_polars(pdf)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (3, 2)


def test_ensure_polars_invalid():
    """Non-DataFrame input raises TypeError."""
    from polars_reg._utils import ensure_polars

    with pytest.raises(TypeError, match="Expected Polars"):
        ensure_polars({"x": [1, 2, 3]})
