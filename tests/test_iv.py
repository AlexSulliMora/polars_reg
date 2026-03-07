import numpy as np
from polars_reg._iv import iv2sls
from polars_reg._ols import ols


def test_iv2sls_basic(iv_data):
    """2SLS should correct endogeneity bias."""
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.model_type == "2SLS"
    # DGP: y = 1.0 + 2.0*x_endog + 0.5*x_exog + u
    np.testing.assert_allclose(result.coefficients[result.names.index("x_endog")], 2.0, atol=0.5)
    np.testing.assert_allclose(result.coefficients[result.names.index("x_exog")], 0.5, atol=0.5)
    assert result.n_obs == 1000


def test_iv2sls_vs_ols_bias(iv_data):
    """OLS on endogenous model should be biased; 2SLS should correct it."""
    ols_result = ols("y ~ x_exog + x_endog", data=iv_data)
    iv_result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    ols_endog = ols_result.coefficients[ols_result.names.index("x_endog")]
    iv_endog = iv_result.coefficients[iv_result.names.index("x_endog")]
    # OLS biased upward (positive corr between x_endog and u)
    assert ols_endog > iv_endog


def test_iv2sls_robust(iv_data):
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data, vcov="HC1")
    assert result.vcov_type == "HC1"
    assert len(result.se) > 0


def test_first_stage_f(iv_data):
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.first_stage_f is not None
    assert result.first_stage_f > 10  # instruments are relevant


def test_iv2sls_summary(iv_data):
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    s = result.summary()
    assert "2SLS" in s
