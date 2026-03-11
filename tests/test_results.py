import numpy as np
import pytest

from polars_reg._results import RegressionResult


def test_result_se():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta,
        vcov=vcov,
        residuals=np.zeros(100),
        names=["x1", "_cons"],
        n_obs=100,
        k=2,
        df_r=98,
        r_squared=0.85,
        r_squared_adj=0.84,
        model_type="OLS",
        vcov_type="iid",
    )
    np.testing.assert_allclose(r.se, [0.1, 0.2])


def test_result_tstat():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta,
        vcov=vcov,
        residuals=np.zeros(100),
        names=["x1", "_cons"],
        n_obs=100,
        k=2,
        df_r=98,
        r_squared=0.85,
        r_squared_adj=0.84,
        model_type="OLS",
        vcov_type="iid",
    )
    np.testing.assert_allclose(r.tstat, [15.0, 10.0])


def test_result_pvalue():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta,
        vcov=vcov,
        residuals=np.zeros(100),
        names=["x1", "_cons"],
        n_obs=100,
        k=2,
        df_r=98,
        r_squared=0.85,
        r_squared_adj=0.84,
        model_type="OLS",
        vcov_type="iid",
    )
    assert all(r.pvalue < 0.001)  # highly significant


def test_result_confint():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta,
        vcov=vcov,
        residuals=np.zeros(100),
        names=["x1", "_cons"],
        n_obs=100,
        k=2,
        df_r=98,
        r_squared=0.85,
        r_squared_adj=0.84,
        model_type="OLS",
        vcov_type="iid",
    )
    ci = r.confint()
    assert ci.shape == (2, 2)
    assert ci[0, 0] < 1.5 < ci[0, 1]
    assert ci[1, 0] < 2.0 < ci[1, 1]


def test_result_summary_returns_string():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta,
        vcov=vcov,
        residuals=np.zeros(100),
        names=["x1", "_cons"],
        n_obs=100,
        k=2,
        df_r=98,
        r_squared=0.85,
        r_squared_adj=0.84,
        model_type="OLS",
        vcov_type="iid",
    )
    s = r.summary()
    assert isinstance(s, str)
    assert "OLS Regression" in s
    assert "x1" in s
    assert "_cons" in s


def test_result_summary_with_fe():
    beta = np.array([1.5])
    vcov = np.diag([0.01])
    r = RegressionResult(
        coefficients=beta,
        vcov=vcov,
        residuals=np.zeros(100),
        names=["x1"],
        n_obs=100,
        k=1,
        df_r=50,
        r_squared=0.85,
        r_squared_adj=0.84,
        model_type="OLS",
        vcov_type="cluster",
        fe_absorbed=["firm_id", "year_id"],
        df_absorbed=68,
        n_clusters={"firm_id": 50},
    )
    s = r.summary()
    assert "Absorbed FE" in s
    assert "Clusters" in s


# ── Additional robustness tests ───────────────────────────────────


def test_confint_invalid_alpha():
    """confint(alpha=0) or confint(alpha=1.5) should produce invalid intervals."""
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta,
        vcov=vcov,
        residuals=np.zeros(100),
        names=["x1", "_cons"],
        n_obs=100,
        k=2,
        df_r=98,
        r_squared=0.85,
        r_squared_adj=0.84,
        model_type="OLS",
        vcov_type="iid",
    )
    # alpha=0 -> t_crit = inf -> interval is [-inf, inf]
    ci_zero = r.confint(alpha=0.0)
    assert ci_zero.shape == (2, 2)
    assert np.all(np.isinf(ci_zero[:, 0]) | np.isinf(ci_zero[:, 1]))

    # alpha=1.5 -> invalid, ppf(1 - 1.5/2) = ppf(0.25) -> narrow interval
    ci_large = r.confint(alpha=1.5)
    assert ci_large.shape == (2, 2)
    # With alpha > 1, the interval should be narrower than the standard 95% CI
    ci_normal = r.confint(alpha=0.05)
    width_large = ci_large[:, 1] - ci_large[:, 0]
    width_normal = ci_normal[:, 1] - ci_normal[:, 0]
    assert np.all(width_large < width_normal)


def test_summary_long_variable_names():
    """Variable names > 20 chars display without error."""
    beta = np.array([1.5, 2.0, -0.3])
    vcov = np.diag([0.01, 0.04, 0.02])
    long_names = [
        "this_is_a_very_long_variable_name",
        "another_super_long_variable_name_here",
        "_cons",
    ]
    r = RegressionResult(
        coefficients=beta,
        vcov=vcov,
        residuals=np.zeros(100),
        names=long_names,
        n_obs=100,
        k=3,
        df_r=97,
        r_squared=0.85,
        r_squared_adj=0.84,
        model_type="OLS",
        vcov_type="iid",
    )
    s = r.summary()
    assert isinstance(s, str)
    assert "this_is_a_very_long_variable_name" in s
    assert "another_super_long_variable_name_here" in s
    assert "_cons" in s
