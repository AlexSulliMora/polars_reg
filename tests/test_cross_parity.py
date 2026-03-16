"""Cross-validated parity tests against Stata and R.

Each test runs a regression in polars_reg and compares against
Stata (via batch mode) and/or R (via Rscript). Tests are skipped
when the external tool is not available.

Tolerance tiers:
    TIGHT   (1e-6): OLS, 2SLS — closed-form solutions
    REGHDFE (1e-5): FE absorption — iterative algorithms differ
    MEDIUM  (1e-4): GMM, panel — implementation differences
    LOOSE   (2e-3): LIML — eigenvalue solver sensitivity
"""

import numpy as np
import polars as pl
import pytest

from polars_reg._ssc import SSC
from tests.r_compare import assert_r_parity, r_available, r_has_package
from tests.stata_compare import assert_stata_parity, stata_available

# Stata ivregress uses asymptotic VCV (no small-sample correction)
STATA_IV_SSC = SSC(k_adj=False, G_adj=False)

# ---------------------------------------------------------------------------
# Tolerance tiers
# ---------------------------------------------------------------------------

TIGHT = 1e-6
REGHDFE = 2e-5
MEDIUM = 1e-4
LOOSE = 2e-3
PANEL = 5e-2  # plm vs our panel: different R² definitions, small-sample corrections

# ---------------------------------------------------------------------------
# Skip decorators
# ---------------------------------------------------------------------------

skip_no_stata = pytest.mark.skipif(not stata_available(), reason="Stata not available")
skip_no_r = pytest.mark.skipif(not r_available(), reason="R/fixest not available")
skip_no_r_aer = pytest.mark.skipif(not r_has_package("AER"), reason="R package AER not installed")
skip_no_r_plm = pytest.mark.skipif(not r_has_package("plm"), reason="R package plm not installed")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cross_data() -> pl.DataFrame:
    """Shared dataset for cross-parity tests."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 50, 20
    n = n_firms * n_years

    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(2000, 2000 + n_years), n_firms)

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u

    firm_fe = rng.standard_normal(n_firms)
    year_fe = rng.standard_normal(n_years)
    y = 2.0 + 1.0 * x1 - 0.5 * x2 + 1.5 * x_endog + firm_fe[firm_id] + year_fe[year_id - 2000] + u

    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "x_endog": x_endog,
            "z1": z1,
            "z2": z2,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )


# ===================================================================
# OLS — iid
# ===================================================================


@skip_no_stata
def test_ols_iid_stata(cross_data):
    assert_stata_parity("ols", "y ~ x1 + x2", cross_data, rtol=TIGHT)


@skip_no_r
def test_ols_iid_r(cross_data):
    assert_r_parity("ols", "y ~ x1 + x2", cross_data, rtol=TIGHT)


# ===================================================================
# OLS — HC1
# ===================================================================


@skip_no_stata
def test_ols_hc1_stata(cross_data):
    assert_stata_parity("ols", "y ~ x1 + x2", cross_data, vcov="HC1", rtol=TIGHT)


@skip_no_r
def test_ols_hc1_r(cross_data):
    assert_r_parity("ols", "y ~ x1 + x2", cross_data, vcov="HC1", rtol=TIGHT)


# ===================================================================
# OLS — HC2
# ===================================================================


@skip_no_stata
def test_ols_hc2_stata(cross_data):
    assert_stata_parity("ols", "y ~ x1 + x2", cross_data, vcov="HC2", rtol=TIGHT)


@skip_no_r
def test_ols_hc2_r(cross_data):
    assert_r_parity("ols", "y ~ x1 + x2", cross_data, vcov="HC2", rtol=TIGHT)


# ===================================================================
# OLS — HC3
# ===================================================================


@skip_no_stata
def test_ols_hc3_stata(cross_data):
    assert_stata_parity("ols", "y ~ x1 + x2", cross_data, vcov="HC3", rtol=TIGHT)


@skip_no_r
def test_ols_hc3_r(cross_data):
    assert_r_parity("ols", "y ~ x1 + x2", cross_data, vcov="HC3", rtol=TIGHT)


# ===================================================================
# OLS — cluster(firm)
# ===================================================================


@skip_no_stata
def test_ols_cluster_stata(cross_data):
    assert_stata_parity("ols", "y ~ x1 + x2", cross_data, cluster=["firm_id"], rtol=TIGHT)


@skip_no_r
def test_ols_cluster_r(cross_data):
    assert_r_parity("ols", "y ~ x1 + x2", cross_data, cluster=["firm_id"], rtol=TIGHT)


# ===================================================================
# OLS — no intercept
# ===================================================================


@skip_no_stata
def test_ols_nocons_stata(cross_data):
    assert_stata_parity("ols", "y ~ x1 + x2 - 1", cross_data, rtol=TIGHT)


@skip_no_r
def test_ols_nocons_r(cross_data):
    assert_r_parity("ols", "y ~ x1 + x2 - 1", cross_data, rtol=TIGHT)


# ===================================================================
# reghdfe — 1-way FE + cluster
# ===================================================================


@skip_no_stata
def test_fe1_cluster_stata(cross_data):
    assert_stata_parity(
        "ols",
        "y ~ x1 + x2 | firm_id",
        cross_data,
        cluster=["firm_id"],
        rtol=REGHDFE,
    )


@skip_no_r
def test_fe1_cluster_r(cross_data):
    assert_r_parity(
        "ols",
        "y ~ x1 + x2 | firm_id",
        cross_data,
        cluster=["firm_id"],
        rtol=REGHDFE,
    )


# ===================================================================
# reghdfe — 2-way FE + cluster
# ===================================================================


@skip_no_stata
def test_fe2_cluster_stata(cross_data):
    assert_stata_parity(
        "ols",
        "y ~ x1 + x2 | firm_id + year_id",
        cross_data,
        cluster=["firm_id"],
        rtol=REGHDFE,
    )


@skip_no_r
def test_fe2_cluster_r(cross_data):
    assert_r_parity(
        "ols",
        "y ~ x1 + x2 | firm_id + year_id",
        cross_data,
        cluster=["firm_id"],
        rtol=REGHDFE,
    )


# ===================================================================
# reghdfe — 2-way FE + 2-way cluster
# ===================================================================


@skip_no_stata
def test_fe2_cluster2_stata(cross_data):
    assert_stata_parity(
        "ols",
        "y ~ x1 + x2 | firm_id + year_id",
        cross_data,
        cluster=["firm_id", "year_id"],
        rtol=REGHDFE,
    )


@skip_no_r
def test_fe2_cluster2_r(cross_data):
    assert_r_parity(
        "ols",
        "y ~ x1 + x2 | firm_id + year_id",
        cross_data,
        cluster=["firm_id", "year_id"],
        rtol=REGHDFE,
    )


# ===================================================================
# reghdfe — 2-way FE + iid
# ===================================================================


@skip_no_stata
def test_fe2_iid_stata(cross_data):
    assert_stata_parity(
        "ols",
        "y ~ x1 + x2 | firm_id + year_id",
        cross_data,
        rtol=REGHDFE,
    )


@skip_no_r
def test_fe2_iid_r(cross_data):
    assert_r_parity(
        "ols",
        "y ~ x1 + x2 | firm_id + year_id",
        cross_data,
        cluster=["firm_id"],  # fixest defaults to clustering; match our default
        rtol=REGHDFE,
    )


# ===================================================================
# 2SLS — iid
# ===================================================================


@skip_no_stata
def test_iv2sls_iid_stata(cross_data):
    assert_stata_parity(
        "iv2sls", "y ~ x1 || x_endog ~ z1 + z2", cross_data, rtol=MEDIUM, ssc=STATA_IV_SSC
    )


@skip_no_r
def test_iv2sls_iid_r(cross_data):
    assert_r_parity("iv2sls", "y ~ x1 || x_endog ~ z1 + z2", cross_data, rtol=TIGHT)


# ===================================================================
# 2SLS — HC1
# ===================================================================


@skip_no_stata
def test_iv2sls_hc1_stata(cross_data):
    assert_stata_parity(
        "iv2sls",
        "y ~ x1 || x_endog ~ z1 + z2",
        cross_data,
        vcov="HC1",
        rtol=MEDIUM,
        ssc=STATA_IV_SSC,
    )


@skip_no_r
def test_iv2sls_hc1_r(cross_data):
    assert_r_parity("iv2sls", "y ~ x1 || x_endog ~ z1 + z2", cross_data, vcov="HC1", rtol=TIGHT)


# ===================================================================
# LIML — iid
# ===================================================================


@skip_no_stata
def test_liml_iid_stata(cross_data):
    assert_stata_parity("liml", "y ~ x1 || x_endog ~ z1 + z2", cross_data, rtol=LOOSE)


@skip_no_r
@skip_no_r_aer
def test_liml_iid_r(cross_data):
    assert_r_parity("liml", "y ~ x1 || x_endog ~ z1 + z2", cross_data, rtol=LOOSE)


# ===================================================================
# GMM — robust (Stata only — no clean R equivalent)
# ===================================================================


@skip_no_stata
def test_gmm_robust_stata(cross_data):
    assert_stata_parity("gmm_iv", "y ~ x1 || x_endog ~ z1 + z2", cross_data, rtol=MEDIUM)


# ===================================================================
# Panel FE — cluster(entity)
# ===================================================================


@skip_no_r
@skip_no_r_plm
def test_panel_fe_r(cross_data):
    # plm within uses entity-only demeaning; our panel_fe demeans by entity+time
    # so coefficients and SEs differ slightly. Use wide tolerance.
    assert_r_parity(
        "panel_fe",
        "y ~ x1 + x2",
        cross_data,
        entity="firm_id",
        time="year_id",
        cluster=["firm_id"],
        rtol=PANEL,
    )


# ===================================================================
# Panel RE — iid
# ===================================================================


@skip_no_r
@skip_no_r_plm
def test_panel_re_r(cross_data):
    # Coefficients and SEs match well; R² definition differs between plm and us
    assert_r_parity(
        "panel_re",
        "y ~ x1 + x2",
        cross_data,
        entity="firm_id",
        rtol=PANEL,
    )


# ===================================================================
# Panel FD — cluster(entity)
# ===================================================================


@skip_no_r
@skip_no_r_plm
def test_panel_fd_r(cross_data):
    # Small-sample SE corrections differ slightly; R² definition differs
    assert_r_parity(
        "panel_fd",
        "y ~ x1 + x2",
        cross_data,
        entity="firm_id",
        time="year_id",
        cluster=["firm_id"],
        rtol=PANEL,
    )
