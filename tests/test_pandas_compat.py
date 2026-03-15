"""Tests that pandas DataFrames are rejected with a clear error message.

polars_reg is Polars-native. Users must call pl.from_pandas(df) before
passing data to any estimator.
"""

import numpy as np
import polars as pl
import pytest

import polars_reg as pr


@pytest.fixture
def pd_df():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "y": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "firm": rng.choice(["A", "B", "C", "D"], n),
        }
    )


def test_ols_rejects_pandas(pd_df):
    with pytest.raises(TypeError, match="pl.from_pandas"):
        pr.ols("y ~ x1 + x2", data=pd_df)


def test_iv2sls_rejects_pandas(pd_df):
    with pytest.raises(TypeError, match="pl.from_pandas"):
        pr.iv2sls("y ~ x1 || x1 ~ x2", data=pd_df)


def test_panel_fe_rejects_pandas(pd_df):
    with pytest.raises(TypeError, match="pl.from_pandas"):
        pr.panel_fe("y ~ x1 + x2", data=pd_df, entity="firm")


def test_from_pandas_then_ols_works(pd_df):
    """Verify the recommended workflow: pl.from_pandas() then estimator."""
    pl_df = pl.from_pandas(pd_df)
    result = pr.ols("y ~ x1 + x2", data=pl_df)
    assert result.n_obs == len(pd_df)
    assert len(result.coefficients) == 3  # intercept + x1 + x2
