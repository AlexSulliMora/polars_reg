"""Tests for the R equivalence module (translation only, no R needed)."""

import pytest

from polars_reg.r_equiv import to_r

# ── to_r: OLS ──────────────────────────────────────────────────────


def test_to_r_ols_simple():
    code = to_r("ols", "y ~ x1 + x2")
    assert "lm(y ~ x1 + x2, data=df)" in code


def test_to_r_ols_robust():
    code = to_r("ols", "y ~ x1 + x2", vcov="HC1")
    assert "library(sandwich)" in code
    assert "vcovHC" in code
    assert '"HC1"' in code


def test_to_r_ols_hc3():
    code = to_r("ols", "y ~ x1 + x2", vcov="HC3")
    assert '"HC3"' in code


def test_to_r_ols_cluster():
    code = to_r("ols", "y ~ x1 + x2", cluster=["firm"])
    assert "library(fixest)" in code
    assert "feols" in code
    assert "vcov=~firm" in code


def test_to_r_ols_cluster_str():
    code = to_r("ols", "y ~ x1 + x2", cluster="firm")
    assert "vcov=~firm" in code


def test_to_r_ols_multiway_cluster():
    code = to_r("ols", "y ~ x1 + x2", cluster=["firm", "year"])
    assert "~firm + year" in code


def test_to_r_ols_noconstant():
    code = to_r("ols", "y ~ x1 + x2 - 1")
    assert "- 1" in code


def test_to_r_feols_fe():
    code = to_r("ols", "y ~ x1 + x2 | fe1 + fe2", cluster=["fe1"])
    assert "library(fixest)" in code
    assert "feols" in code
    assert "| fe1 + fe2" in code
    assert "vcov=~fe1" in code


def test_to_r_feols_fe_iid():
    code = to_r("ols", "y ~ x1 | fe1")
    assert "feols" in code
    assert '"iid"' in code


# ── to_r: IV ───────────────────────────────────────────────────────


def test_to_r_iv2sls():
    code = to_r("iv2sls", "y ~ x_exog || x_endog ~ z1 + z2")
    assert "library(fixest)" in code
    assert "feols" in code
    assert "x_endog ~ z1 + z2" in code


def test_to_r_iv2sls_with_fe():
    code = to_r("iv2sls", "y ~ x_exog | fe1 | x_endog ~ z1 + z2")
    assert "| fe1 |" in code
    assert "x_endog ~ z1 + z2" in code


def test_to_r_liml():
    code = to_r("liml", "y ~ x_exog || x_endog ~ z1 + z2")
    assert "library(AER)" in code
    assert "ivreg" in code
    assert '"liml"' in code


def test_to_r_gmm():
    code = to_r("gmm_iv", "y ~ x_exog || x_endog ~ z1 + z2")
    assert "No direct single-function" in code
    assert "feols" in code


# ── to_r: Panel ────────────────────────────────────────────────────


def test_to_r_panel_fe():
    code = to_r("panel_fe", "y ~ x1 + x2", entity="firm", time="year")
    assert "library(plm)" in code
    assert 'model="within"' in code
    assert '"firm"' in code
    assert '"year"' in code


def test_to_r_panel_fe_cluster():
    code = to_r("panel_fe", "y ~ x1 + x2", entity="firm", cluster=["firm"])
    assert "vcovHC" in code


def test_to_r_panel_re():
    code = to_r("panel_re", "y ~ x1 + x2", entity="firm")
    assert 'model="random"' in code


def test_to_r_panel_fd():
    code = to_r("panel_fd", "y ~ x1 + x2", entity="firm", time="year")
    assert 'model="fd"' in code


# ── to_r: errors ──────────────────────────────────────────────────


def test_to_r_unknown_estimator():
    with pytest.raises(ValueError, match="Unknown estimator"):
        to_r("bad_estimator", "y ~ x1")


def test_to_r_panel_fe_no_entity():
    with pytest.raises(ValueError, match="entity"):
        to_r("panel_fe", "y ~ x1")


def test_to_r_panel_fd_no_time():
    with pytest.raises(ValueError, match="time"):
        to_r("panel_fd", "y ~ x1", entity="firm")


# ── to_r: indicators and interactions ─────────────────────────────


def test_to_r_ols_indicator():
    """Indicator variables should use factor() in R."""
    code = to_r("ols", "y ~ x1 + i.group")
    assert "factor(group)" in code


def test_to_r_ols_interaction():
    """Interaction terms should use : in R."""
    code = to_r("ols", "y ~ x1 + x1:x2")
    assert "x1:x2" in code


def test_to_r_ols_indicator_interaction():
    """Indicator interacted with continuous should use factor():x in R."""
    code = to_r("ols", "y ~ x1 + i.group:x2")
    assert "factor(group):x2" in code


def test_to_r_feols_fe_noconstant():
    """FE model with -1 should pass through - 1."""
    code = to_r("ols", "y ~ x1 - 1 | fe1", cluster=["fe1"])
    assert "- 1" in code
    assert "feols" in code


# ── compare_r: smoke test (no rpy2) ──────────────────────────────


def test_compare_r_no_rpy2(simple_data, capsys):
    """compare_r should work even without rpy2, printing polars results."""
    from polars_reg.r_equiv import compare_r

    report = compare_r("ols", "y ~ x1 + x2", data=simple_data)
    assert report.match is None  # rpy2 not available
    assert report.polars_coefs is not None
    assert "polars_reg vs R" in report.summary()
    captured = capsys.readouterr()
    assert "polars_reg vs R" in captured.out
