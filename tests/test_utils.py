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
