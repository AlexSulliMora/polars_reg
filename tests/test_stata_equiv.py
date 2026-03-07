"""Tests for the public stata equivalence module."""

import pytest

from polars_reg.stata import to_stata

# ── to_stata: OLS ──────────────────────────────────────────────────


def test_to_stata_ols_simple():
    cmd = to_stata("ols", "y ~ x1 + x2")
    assert cmd == "reg y x1 x2"


def test_to_stata_ols_robust():
    cmd = to_stata("ols", "y ~ x1 + x2", vcov="HC1")
    assert cmd == "reg y x1 x2, vce(robust)"


def test_to_stata_ols_cluster():
    cmd = to_stata("ols", "y ~ x1 + x2", cluster=["firm"])
    assert cmd == "reg y x1 x2, vce(cluster firm)"


def test_to_stata_ols_cluster_str():
    cmd = to_stata("ols", "y ~ x1 + x2", cluster="firm")
    assert cmd == "reg y x1 x2, vce(cluster firm)"


def test_to_stata_ols_noconstant():
    cmd = to_stata("ols", "y ~ x1 + x2 - 1")
    assert "noconstant" in cmd


def test_to_stata_reghdfe():
    cmd = to_stata("ols", "y ~ x1 + x2 | fe1 + fe2", cluster=["fe1"])
    assert cmd.startswith("reghdfe")
    assert "absorb(fe1 fe2)" in cmd
    assert "vce(cluster fe1)" in cmd


def test_to_stata_reghdfe_multiway():
    cmd = to_stata("ols", "y ~ x1 | fe1 + fe2", cluster=["fe1", "fe2"])
    assert "vce(cluster fe1 fe2)" in cmd


# ── to_stata: IV ──────────────────────────────────────────────────


def test_to_stata_iv2sls():
    cmd = to_stata("iv2sls", "y ~ x_exog || x_endog ~ z1 + z2")
    assert "ivregress 2sls" in cmd
    assert "(x_endog = z1 z2)" in cmd
    assert "small" in cmd


def test_to_stata_liml():
    cmd = to_stata("liml", "y ~ x_exog || x_endog ~ z1 + z2")
    assert "ivregress liml" in cmd


def test_to_stata_gmm():
    cmd = to_stata("gmm_iv", "y ~ x_exog || x_endog ~ z1 + z2")
    assert "ivregress gmm" in cmd
    assert "wmatrix(robust)" in cmd


def test_to_stata_iv_with_fe():
    cmd = to_stata("iv2sls", "y ~ x_exog | fe1 | x_endog ~ z1 + z2")
    assert "ivreghdfe" in cmd
    assert "absorb(fe1)" in cmd


# ── to_stata: Panel ────────────────────────────────────────────────


def test_to_stata_panel_fe():
    cmd = to_stata("panel_fe", "y ~ x1 + x2", entity="firm", time="year")
    assert "xtset firm year" in cmd
    assert "xtreg" in cmd
    assert ", fe" in cmd


def test_to_stata_panel_fe_cluster():
    cmd = to_stata("panel_fe", "y ~ x1 + x2", entity="firm", cluster=["firm"])
    assert "vce(cluster firm)" in cmd


def test_to_stata_panel_re():
    cmd = to_stata("panel_re", "y ~ x1 + x2", entity="firm")
    assert "xtset firm" in cmd
    assert ", re" in cmd


def test_to_stata_panel_fd():
    cmd = to_stata("panel_fd", "y ~ x1 + x2", entity="firm", time="year")
    assert "xtset firm year" in cmd
    assert "D.y" in cmd
    assert "D.x1" in cmd
    assert "D.x2" in cmd


# ── to_stata: pystata wrapper ─────────────────────────────────────


def test_to_stata_pystata():
    cmd = to_stata("ols", "y ~ x1 + x2", pystata=True)
    assert "import stata_setup" in cmd
    assert 'stata.run("reg y x1 x2")' in cmd
    assert "matrix list e(b)" in cmd


# ── to_stata: errors ──────────────────────────────────────────────


def test_to_stata_unknown_estimator():
    with pytest.raises(ValueError, match="Unknown estimator"):
        to_stata("bad_estimator", "y ~ x1")


def test_to_stata_panel_fe_no_entity():
    with pytest.raises(ValueError, match="entity"):
        to_stata("panel_fe", "y ~ x1")


def test_to_stata_panel_fd_no_time():
    with pytest.raises(ValueError, match="time"):
        to_stata("panel_fd", "y ~ x1", entity="firm")


# ── compare_stata: smoke test (no pystata) ─────────────────────────


def test_compare_stata_no_pystata(simple_data, capsys):
    """compare_stata should work even without pystata, printing polars results."""
    from polars_reg.stata import compare_stata

    report = compare_stata("ols", "y ~ x1 + x2", data=simple_data)
    assert report.match is None  # pystata not available
    assert report.polars_coefs is not None
    assert "polars_reg vs Stata" in report.summary()
    captured = capsys.readouterr()
    assert "polars_reg vs Stata" in captured.out
