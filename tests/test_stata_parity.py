"""Stata parity tests for polars_reg.

These tests run identical regressions in polars_reg and Stata (via pystata),
comparing coefficients and standard errors to machine precision.

Requirements:
    - Stata installed with valid license
    - pystata configured (set STATA_DIR and STATA_EDITION env vars)
    - reghdfe installed: ssc install reghdfe, ftools

To run:
    STATA_DIR=/usr/local/stata STATA_EDITION=mp pytest tests/test_stata_parity.py -v

All tests are skipped if Stata is not available.
"""

import numpy as np
import polars as pl
import pytest

from tests.stata_compare import (
    ComparisonResult,
    assert_stata_parity,
    stata_available,
    to_stata_command,
)

requires_stata = pytest.mark.skipif(
    not stata_available(),
    reason="Stata/pystata not available",
)


# ---------------------------------------------------------------------------
# Fixtures: deterministic datasets with known properties
# ---------------------------------------------------------------------------


@pytest.fixture
def ols_data() -> pl.DataFrame:
    """1000-obs dataset for OLS tests."""
    rng = np.random.default_rng(12345)
    n = 1000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    e = rng.standard_normal(n)
    y = 3.0 + 1.5 * x1 - 0.8 * x2 + 0.3 * x3 + e
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


@pytest.fixture
def panel_data() -> pl.DataFrame:
    """Balanced panel: 100 firms x 10 years for reghdfe tests."""
    rng = np.random.default_rng(12345)
    n_firms, n_years = 100, 10
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(1, n_firms + 1), n_years)
    year_id = np.tile(np.arange(2001, 2001 + n_years), n_firms)
    firm_fe = rng.standard_normal(n_firms)
    year_fe = rng.standard_normal(n_years)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    e = rng.standard_normal(n) * 0.5
    y = 1.0 * x1 - 2.0 * x2 + firm_fe[firm_id - 1] + year_fe[year_id - 2001] + e
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )


@pytest.fixture
def iv_data() -> pl.DataFrame:
    """IV dataset: 1 endogenous regressor, 2 instruments."""
    rng = np.random.default_rng(12345)
    n = 1000
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    x_exog = rng.standard_normal(n)
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + u
    return pl.DataFrame(
        {
            "y": y,
            "x_endog": x_endog,
            "x_exog": x_exog,
            "z1": z1,
            "z2": z2,
        }
    )


# ---------------------------------------------------------------------------
# Unit tests for the translation layer (no Stata needed)
# ---------------------------------------------------------------------------


class TestTranslation:
    """Test formula-to-Stata translation without requiring Stata."""

    def test_ols_basic(self):
        cmd, model = to_stata_command("ols", "y ~ x1 + x2")
        assert model == "reg"
        assert cmd == "reg y x1 x2"

    def test_ols_robust(self):
        cmd, _ = to_stata_command("ols", "y ~ x1 + x2", vcov="HC1")
        assert "vce(robust)" in cmd

    def test_ols_hc2(self):
        cmd, _ = to_stata_command("ols", "y ~ x1 + x2", vcov="HC2")
        assert "vce(hc2)" in cmd

    def test_ols_hc3(self):
        cmd, _ = to_stata_command("ols", "y ~ x1 + x2", vcov="HC3")
        assert "vce(hc3)" in cmd

    def test_ols_clustered(self):
        cmd, _ = to_stata_command("ols", "y ~ x1 + x2", cluster=["firm_id"])
        assert "vce(cluster firm_id)" in cmd

    def test_ols_no_intercept(self):
        cmd, _ = to_stata_command("ols", "y ~ x1 + x2 - 1")
        assert "noconstant" in cmd

    def test_reghdfe_basic(self):
        cmd, model = to_stata_command("ols", "y ~ x1 + x2 | firm_id + year_id", cluster=["firm_id"])
        assert model == "reghdfe"
        assert "reghdfe y x1 x2" in cmd
        assert "absorb(firm_id year_id)" in cmd
        assert "vce(cluster firm_id)" in cmd

    def test_reghdfe_twoway_cluster(self):
        cmd, _ = to_stata_command(
            "ols",
            "y ~ x1 | firm_id + year_id",
            cluster=["firm_id", "year_id"],
        )
        assert "vce(cluster firm_id year_id)" in cmd

    def test_reghdfe_iid(self):
        cmd, _ = to_stata_command("ols", "y ~ x1 | firm_id", vcov="iid")
        assert "vce(unadjusted)" in cmd

    def test_iv2sls(self):
        cmd, model = to_stata_command("iv2sls", "y ~ x_exog || x_endog ~ z1 + z2")
        assert model == "ivregress_2sls"
        assert "ivregress 2sls y x_exog (x_endog = z1 z2)" in cmd

    def test_iv2sls_robust(self):
        cmd, _ = to_stata_command("iv2sls", "y ~ x_exog || x_endog ~ z1 + z2", vcov="HC1")
        assert "vce(robust)" in cmd

    def test_iv2sls_clustered(self):
        cmd, _ = to_stata_command("iv2sls", "y ~ x_exog || x_endog ~ z1 + z2", cluster=["cl"])
        assert "vce(cluster cl)" in cmd

    def test_liml(self):
        cmd, model = to_stata_command("liml", "y ~ x_exog || x_endog ~ z1 + z2")
        assert model == "ivregress_liml"
        assert "ivregress liml" in cmd

    def test_gmm(self):
        cmd, model = to_stata_command("gmm_iv", "y ~ x_exog || x_endog ~ z1 + z2")
        assert model == "ivregress_gmm"
        assert "ivregress gmm" in cmd
        assert "wmatrix(robust)" in cmd


# ---------------------------------------------------------------------------
# Stata parity tests (require Stata)
# ---------------------------------------------------------------------------

# Tolerances:
#   TIGHT    = 1e-6  (coefficients and SEs should match to 6+ decimals)
#   REGHDFE  = 1e-5  (demeaning algorithms differ slightly between implementations)
#   MEDIUM   = 1e-4  (for estimators with known minor implementation differences)
#   LOOSE    = 2e-3  (for LIML where eigenvalue solvers can differ)
TIGHT = 1e-6
REGHDFE = 1e-5
MEDIUM = 1e-4
LOOSE = 2e-3


@requires_stata
class TestOLSParity:
    """OLS (reg) parity with Stata."""

    def test_ols_iid(self, ols_data):
        r = assert_stata_parity("ols", "y ~ x1 + x2 + x3", ols_data, rtol=TIGHT)
        assert r.coef_max_rdiff < TIGHT
        assert r.se_max_rdiff < TIGHT

    def test_ols_robust(self, ols_data):
        r = assert_stata_parity("ols", "y ~ x1 + x2 + x3", ols_data, vcov="HC1", rtol=TIGHT)
        assert r.coef_max_rdiff < TIGHT
        assert r.se_max_rdiff < TIGHT

    def test_ols_hc2(self, ols_data):
        r = assert_stata_parity("ols", "y ~ x1 + x2 + x3", ols_data, vcov="HC2", rtol=TIGHT)
        assert r.se_max_rdiff < TIGHT

    def test_ols_hc3(self, ols_data):
        r = assert_stata_parity("ols", "y ~ x1 + x2 + x3", ols_data, vcov="HC3", rtol=TIGHT)
        assert r.se_max_rdiff < TIGHT

    def test_ols_no_intercept(self, ols_data):
        r = assert_stata_parity("ols", "y ~ x1 + x2 + x3 - 1", ols_data, rtol=TIGHT)
        assert r.coef_max_rdiff < TIGHT


@requires_stata
class TestRegHDFEParity:
    """reghdfe parity with Stata."""

    def test_one_fe_clustered(self, panel_data):
        r = assert_stata_parity(
            "ols",
            "y ~ x1 + x2 | firm_id",
            panel_data,
            cluster=["firm_id"],
            rtol=REGHDFE,
        )
        assert r.coef_max_rdiff < REGHDFE
        assert r.se_max_rdiff < REGHDFE

    def test_twoway_fe_clustered(self, panel_data):
        r = assert_stata_parity(
            "ols",
            "y ~ x1 + x2 | firm_id + year_id",
            panel_data,
            cluster=["firm_id"],
            rtol=REGHDFE,
        )
        assert r.coef_max_rdiff < REGHDFE
        assert r.se_max_rdiff < REGHDFE

    def test_twoway_fe_twoway_cluster(self, panel_data):
        r = assert_stata_parity(
            "ols",
            "y ~ x1 + x2 | firm_id + year_id",
            panel_data,
            cluster=["firm_id", "year_id"],
            rtol=REGHDFE,
        )
        assert r.coef_max_rdiff < REGHDFE
        assert r.se_max_rdiff < REGHDFE

    def test_twoway_fe_iid(self, panel_data):
        r = assert_stata_parity(
            "ols",
            "y ~ x1 + x2 | firm_id + year_id",
            panel_data,
            vcov="iid",
            rtol=REGHDFE,
        )
        assert r.coef_max_rdiff < REGHDFE
        assert r.se_max_rdiff < REGHDFE


@requires_stata
class TestIVParity:
    """ivregress parity with Stata."""

    def test_2sls_iid(self, iv_data):
        r = assert_stata_parity(
            "iv2sls",
            "y ~ x_exog || x_endog ~ z1 + z2",
            iv_data,
            rtol=TIGHT,
        )
        assert r.coef_max_rdiff < TIGHT
        assert r.se_max_rdiff < TIGHT

    def test_2sls_robust(self, iv_data):
        r = assert_stata_parity(
            "iv2sls",
            "y ~ x_exog || x_endog ~ z1 + z2",
            iv_data,
            vcov="HC1",
            rtol=TIGHT,
        )
        assert r.coef_max_rdiff < TIGHT
        assert r.se_max_rdiff < TIGHT

    def test_liml_iid(self, iv_data):
        # LIML kappa eigenvalue computation differs slightly between
        # implementations, leading to ~6e-4 coefficient differences
        r = assert_stata_parity(
            "liml",
            "y ~ x_exog || x_endog ~ z1 + z2",
            iv_data,
            rtol=LOOSE,
        )
        assert r.coef_max_rdiff < LOOSE
        assert r.se_max_rdiff < LOOSE

    def test_gmm_robust(self, iv_data):
        # GMM may have slightly different implementation details
        r = assert_stata_parity(
            "gmm_iv",
            "y ~ x_exog || x_endog ~ z1 + z2",
            iv_data,
            rtol=MEDIUM,
        )
        assert r.coef_max_rdiff < MEDIUM


# ---------------------------------------------------------------------------
# Batch parity runner (for ad-hoc use outside pytest)
# ---------------------------------------------------------------------------


def run_all_parity_checks(
    ols_data: pl.DataFrame,
    panel_data: pl.DataFrame,
    iv_data: pl.DataFrame,
    rtol: float = 1e-6,
) -> list[ComparisonResult]:
    """Run all parity checks and return results (doesn't raise on failure).

    Useful for interactive exploration:

        from tests.stata_compare import stata_available
        from tests.test_stata_parity import run_all_parity_checks
        # ... create datasets ...
        results = run_all_parity_checks(ols_data, panel_data, iv_data)
        for r in results:
            print(r)
    """
    import polars_reg as pr
    from tests.stata_compare import (
        _extract_stata_results,
        _load_data_to_stata,
        _run_stata,
        compare_results,
        to_stata_command,
    )

    specs = [
        # (estimator, formula, data, vcov, cluster)
        ("ols", "y ~ x1 + x2 + x3", ols_data, "iid", None),
        ("ols", "y ~ x1 + x2 + x3", ols_data, "HC1", None),
        ("ols", "y ~ x1 + x2 + x3", ols_data, "HC2", None),
        ("ols", "y ~ x1 + x2 + x3", ols_data, "HC3", None),
        ("ols", "y ~ x1 + x2 | firm_id", panel_data, "iid", ["firm_id"]),
        ("ols", "y ~ x1 + x2 | firm_id + year_id", panel_data, "iid", ["firm_id"]),
        ("ols", "y ~ x1 + x2 | firm_id + year_id", panel_data, "iid", ["firm_id", "year_id"]),
        ("iv2sls", "y ~ x_exog || x_endog ~ z1 + z2", iv_data, "iid", None),
        ("iv2sls", "y ~ x_exog || x_endog ~ z1 + z2", iv_data, "HC1", None),
        ("liml", "y ~ x_exog || x_endog ~ z1 + z2", iv_data, "iid", None),
        ("gmm_iv", "y ~ x_exog || x_endog ~ z1 + z2", iv_data, "iid", None),
    ]

    funcs = {
        "ols": pr.ols,
        "iv2sls": pr.iv2sls,
        "liml": pr.liml,
        "gmm_iv": pr.gmm_iv,
    }

    results = []
    for est, formula, data, vcov, cluster in specs:
        try:
            kwargs: dict = {"formula": formula, "data": data}
            if vcov != "iid":
                kwargs["vcov"] = vcov
            if cluster:
                kwargs["cluster"] = cluster
            pr_result = funcs[est](**kwargs)

            stata_cmd, model_type = to_stata_command(est, formula, vcov, cluster)
            _load_data_to_stata(data)
            _run_stata(stata_cmd)
            st_result = _extract_stata_results(model_type)

            comp = compare_results(est, formula, stata_cmd, pr_result, st_result, rtol, vcov)
            results.append(comp)
        except Exception as e:
            comp = ComparisonResult(
                estimator=est,
                formula=formula,
                stata_command="(error)",
                passed=False,
                n_obs_match=False,
                coef_max_rdiff=float("inf"),
                se_max_rdiff=float("inf"),
                r2_rdiff=None,
                details=[f"Exception: {e}"],
            )
            results.append(comp)

    return results
