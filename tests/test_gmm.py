import numpy as np

from polars_reg._gmm import gmm_iv, liml


def test_liml_basic(iv_data):
    result = liml("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.model_type == "LIML"
    np.testing.assert_allclose(result.coefficients[result.names.index("x_endog")], 2.0, atol=0.5)
    assert result.n_obs == 1000


def test_liml_vs_ols(iv_data):
    """LIML should correct endogeneity bias like 2SLS."""
    from polars_reg._ols import ols

    ols_result = ols("y ~ x_exog + x_endog", data=iv_data)
    liml_result = liml("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    ols_endog = ols_result.coefficients[ols_result.names.index("x_endog")]
    liml_endog = liml_result.coefficients[liml_result.names.index("x_endog")]
    assert abs(liml_endog - 2.0) < abs(ols_endog - 2.0)


def test_gmm_basic(iv_data):
    result = gmm_iv("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.model_type == "GMM"
    np.testing.assert_allclose(result.coefficients[result.names.index("x_endog")], 2.0, atol=0.5)


def test_gmm_j_stat(iv_data):
    """With 2 instruments and 1 endogenous var, Hansen J should be available."""
    result = gmm_iv("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.j_stat is not None
    assert result.j_pvalue is not None
    assert result.j_pvalue > 0.05  # should not reject at 5% level (valid instruments in DGP)


def test_gmm_summary(iv_data):
    result = gmm_iv("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    s = result.summary()
    assert "GMM" in s
    assert "Hansen J" in s


def test_liml_nw(iv_data_panel):
    """LIML with Newey-West SEs."""
    from polars_reg._gmm import liml

    result = liml(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"
    assert all(se > 0 for se in result.se)


def test_liml_dk(iv_data_panel):
    """LIML with Driscoll-Kraay SEs."""
    from polars_reg._gmm import liml

    result = liml(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"


def test_gmm_nw(iv_data_panel):
    """GMM with Newey-West SEs."""
    from polars_reg._gmm import gmm_iv

    result = gmm_iv(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"


def test_gmm_dk(iv_data_panel):
    """GMM with Driscoll-Kraay SEs."""
    from polars_reg._gmm import gmm_iv

    result = gmm_iv(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"
