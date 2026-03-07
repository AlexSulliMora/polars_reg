import numpy as np
import polars as pl

from polars_reg._ols import ols


def test_ols_basic(simple_data):
    result = ols("y ~ x1 + x2", data=simple_data)
    # DGP: y = 2.0 + 1.5*x1 - 0.5*x2 + N(0, 0.25)
    assert result.n_obs == 1000
    assert result.model_type == "OLS"
    np.testing.assert_allclose(result.coefficients[0], 1.5, atol=0.1)   # x1
    np.testing.assert_allclose(result.coefficients[1], -0.5, atol=0.1)  # x2
    np.testing.assert_allclose(result.coefficients[2], 2.0, atol=0.1)   # _cons
    assert result.r_squared > 0.8
    assert result.names == ["x1", "x2", "_cons"]


def test_ols_robust(simple_data):
    result = ols("y ~ x1 + x2", data=simple_data, vcov="HC1")
    assert result.vcov_type == "HC1"
    assert len(result.se) == 3
    # Robust SEs should be close to iid SEs for homoskedastic data
    iid_result = ols("y ~ x1 + x2", data=simple_data)
    np.testing.assert_allclose(result.se, iid_result.se, rtol=0.2)


def test_ols_clustered(panel_data):
    result = ols("y ~ x1 + x2", data=panel_data, cluster=["firm_id"])
    assert result.vcov_type == "cluster"
    assert result.n_clusters == {"firm_id": 50}
    assert result.df_r == 49


def test_ols_twoway_clustered(panel_data):
    result = ols("y ~ x1 + x2", data=panel_data, cluster=["firm_id", "year_id"])
    assert result.vcov_type == "cluster"
    assert result.n_clusters == {"firm_id": 50, "year_id": 20}
    assert result.df_r == 19  # min(50, 20) - 1


def test_ols_summary(simple_data):
    result = ols("y ~ x1 + x2", data=simple_data)
    s = result.summary()
    assert "OLS Regression" in s
    assert "x1" in s
    assert "x2" in s
    assert "_cons" in s


def test_ols_no_intercept(simple_data):
    result = ols("y ~ x1 + x2 - 1", data=simple_data)
    assert "_cons" not in result.names
    assert len(result.coefficients) == 2
