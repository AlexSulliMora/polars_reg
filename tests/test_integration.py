"""Integration tests: end-to-end usage of the public API."""
import numpy as np
import polars as pl
import pytest

import polars_reg


def test_ols_from_package(simple_data):
    """Test that ols is accessible from package root."""
    result = polars_reg.ols("y ~ x1 + x2", data=simple_data)
    assert result.model_type == "OLS"
    assert result.n_obs == 1000


def test_ols_with_fe_from_package(panel_data):
    result = polars_reg.ols("y ~ x1 + x2 | firm_id + year_id", data=panel_data, cluster=["firm_id"])
    assert result.fe_absorbed == ["firm_id", "year_id"]
    np.testing.assert_allclose(result.coefficients[0], 1.0, atol=0.15)


def test_result_object_api(simple_data):
    """Test that RegressionResult has expected properties."""
    result = polars_reg.ols("y ~ x1 + x2", data=simple_data)
    assert hasattr(result, "se")
    assert hasattr(result, "tstat")
    assert hasattr(result, "pvalue")
    assert hasattr(result, "confint")
    assert hasattr(result, "summary")
    ci = result.confint()
    assert ci.shape == (3, 2)
    s = result.summary()
    assert isinstance(s, str)


def test_ols_lazyframe(simple_data):
    """Should accept LazyFrame as well."""
    lazy = simple_data.lazy()
    result = polars_reg.ols("y ~ x1 + x2", data=lazy)
    assert result.n_obs == 1000


@pytest.mark.skipif(not hasattr(polars_reg, "iv2sls"), reason="iv2sls not yet available")
def test_iv2sls_from_package(iv_data):
    result = polars_reg.iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.model_type == "2SLS"


@pytest.mark.skipif(not hasattr(polars_reg, "gmm_iv"), reason="gmm_iv not yet available")
def test_gmm_from_package(iv_data):
    result = polars_reg.gmm_iv("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.model_type == "GMM"


@pytest.mark.skipif(not hasattr(polars_reg, "panel_fe"), reason="panel_fe not yet available")
def test_panel_fe_from_package(panel_data):
    result = polars_reg.panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id")
    assert result.model_type == "Panel FE"
