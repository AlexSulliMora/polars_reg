"""Tests for probit and logit (binary choice) models."""

import numpy as np
import polars as pl
import pytest
from scipy import stats

from polars_reg import logit, marginal_effects, odds_ratios, probit


@pytest.fixture
def binary_data():
    """Binary outcome dataset."""
    rng = np.random.default_rng(42)
    n = 2000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # True probit: P(y=1) = Phi(0.5 + 1.0*x1 - 0.5*x2)
    xb = 0.5 + 1.0 * x1 - 0.5 * x2
    prob = stats.norm.cdf(xb)
    y = (rng.uniform(size=n) < prob).astype(float)
    firm = np.repeat(np.arange(100), 20)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm})


@pytest.fixture
def logit_data():
    """Binary outcome dataset for logit."""
    rng = np.random.default_rng(42)
    n = 2000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    xb = 0.5 + 1.0 * x1 - 0.5 * x2
    prob = 1.0 / (1.0 + np.exp(-xb))
    y = (rng.uniform(size=n) < prob).astype(float)
    firm = np.repeat(np.arange(100), 20)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm})


def test_probit_basic(binary_data):
    """Probit runs and recovers approximate coefficients."""
    r = probit("y ~ x1 + x2", data=binary_data)
    assert r.model_type == "Probit"
    assert r.n_obs == 2000
    # Check coefficient signs match DGP
    idx_x1 = r.names.index("x1")
    idx_x2 = r.names.index("x2")
    assert r.coefficients[idx_x1] > 0  # true: +1.0
    assert r.coefficients[idx_x2] < 0  # true: -0.5


def test_probit_coefficients_close(binary_data):
    """Probit coefficients should be close to true values with large N."""
    r = probit("y ~ x1 + x2", data=binary_data)
    idx_x1 = r.names.index("x1")
    idx_x2 = r.names.index("x2")
    idx_cons = r.names.index("_cons")
    # With n=2000, should be within ~0.2 of true values
    assert abs(r.coefficients[idx_cons] - 0.5) < 0.2
    assert abs(r.coefficients[idx_x1] - 1.0) < 0.2
    assert abs(r.coefficients[idx_x2] - (-0.5)) < 0.2


def test_probit_se_positive(binary_data):
    """SEs should be positive."""
    r = probit("y ~ x1 + x2", data=binary_data)
    assert np.all(r.se > 0)


def test_probit_pseudo_r2(binary_data):
    """Pseudo R² should be between 0 and 1."""
    r = probit("y ~ x1 + x2", data=binary_data)
    assert 0 < r.r_squared < 1


def test_probit_summary(binary_data):
    """Probit summary displays correctly."""
    r = probit("y ~ x1 + x2", data=binary_data)
    s = r.summary()
    assert "Probit" in s
    assert "x1" in s


def test_probit_robust(binary_data):
    """Probit with robust SEs."""
    r = probit("y ~ x1 + x2", data=binary_data, vcov="HC1")
    assert r.vcov_type == "HC1"
    assert np.all(r.se > 0)


def test_probit_clustered(binary_data):
    """Probit with clustered SEs."""
    r = probit("y ~ x1 + x2", data=binary_data, cluster="firm")
    assert r.vcov_type == "cluster"
    assert np.all(r.se > 0)


def test_probit_no_fe(binary_data):
    """Probit with FE raises error."""
    with pytest.raises(ValueError, match="does not support"):
        probit("y ~ x1 + x2 | firm", data=binary_data)


def test_probit_non_binary_error():
    """Non-binary y raises error."""
    df = pl.DataFrame({"y": [0.0, 0.5, 1.0], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="binary"):
        probit("y ~ x", data=df)


def test_logit_basic(logit_data):
    """Logit runs and recovers approximate coefficients."""
    r = logit("y ~ x1 + x2", data=logit_data)
    assert r.model_type == "Logit"
    idx_x1 = r.names.index("x1")
    idx_x2 = r.names.index("x2")
    assert r.coefficients[idx_x1] > 0
    assert r.coefficients[idx_x2] < 0


def test_logit_coefficients_close(logit_data):
    """Logit coefficients should be close to true values."""
    r = logit("y ~ x1 + x2", data=logit_data)
    idx_x1 = r.names.index("x1")
    idx_x2 = r.names.index("x2")
    idx_cons = r.names.index("_cons")
    assert abs(r.coefficients[idx_cons] - 0.5) < 0.2
    assert abs(r.coefficients[idx_x1] - 1.0) < 0.2
    assert abs(r.coefficients[idx_x2] - (-0.5)) < 0.2


def test_logit_robust(logit_data):
    """Logit with robust SEs."""
    r = logit("y ~ x1 + x2", data=logit_data, vcov="HC1")
    assert r.vcov_type == "HC1"
    assert np.all(r.se > 0)


def test_logit_clustered(logit_data):
    """Logit with clustered SEs."""
    r = logit("y ~ x1 + x2", data=logit_data, cluster="firm")
    assert r.vcov_type == "cluster"
    assert np.all(r.se > 0)


def test_logit_pseudo_r2(logit_data):
    """Logit pseudo R² is between 0 and 1."""
    r = logit("y ~ x1 + x2", data=logit_data)
    assert 0 < r.r_squared < 1


def test_marginal_effects_at_mean(binary_data):
    """Marginal effects at the mean for probit."""
    r = probit("y ~ x1 + x2", data=binary_data)
    me = marginal_effects(r, at="mean")
    assert "dy_dx" in me.columns
    assert "se" in me.columns
    assert me.shape[0] == 3  # x1, x2, _cons
    # MEs should be smaller than coefficients (scaled by density)
    idx_x1 = r.names.index("x1")
    assert abs(me["dy_dx"][idx_x1]) < abs(r.coefficients[idx_x1])


def test_marginal_effects_average(binary_data):
    """Average marginal effects for probit."""
    r = probit("y ~ x1 + x2", data=binary_data)
    me = marginal_effects(r, at="average")
    assert me.shape[0] == 3
    assert np.all(me["se"].to_numpy() > 0)


def test_marginal_effects_logit(logit_data):
    """Marginal effects work for logit too."""
    r = logit("y ~ x1 + x2", data=logit_data)
    me = marginal_effects(r, at="mean")
    assert me.shape[0] == 3
    idx_x1 = r.names.index("x1")
    # AME for logit: Lambda(1-Lambda)*beta, should be positive for x1
    assert me["dy_dx"][idx_x1] > 0


def test_marginal_effects_wrong_model(binary_data):
    """Marginal effects on OLS raises error."""
    from polars_reg import ols

    r = ols("y ~ x1 + x2", data=binary_data)
    with pytest.raises(ValueError, match="Probit or Logit"):
        marginal_effects(r)


def test_logit_coef_table(logit_data):
    """Logit coef_table returns valid DataFrame."""
    r = logit("y ~ x1 + x2", data=logit_data)
    ct = r.coef_table()
    assert ct.shape[0] == 3
    assert "coef" in ct.columns
    assert "se" in ct.columns
    assert "p" in ct.columns


def test_odds_ratios_basic(logit_data):
    """Odds ratios from logit: exp(beta), with delta-method SE."""
    r = logit("y ~ x1 + x2", data=logit_data)
    ordf = odds_ratios(r)
    assert "or" in ordf.columns
    assert "se" in ordf.columns
    assert "ci_lower" in ordf.columns
    assert "ci_upper" in ordf.columns
    assert ordf.shape[0] == 3  # x1, x2, _cons
    # OR = exp(beta), so all should be positive
    assert np.all(ordf["or"].to_numpy() > 0)
    # OR for x1 should be > 1 (positive coefficient)
    idx_x1 = r.names.index("x1")
    assert ordf["or"][idx_x1] > 1.0


def test_odds_ratios_exp_beta(logit_data):
    """Odds ratios equal exp(coefficients)."""
    r = logit("y ~ x1 + x2", data=logit_data)
    ordf = odds_ratios(r)
    expected_or = np.exp(r.coefficients)
    np.testing.assert_allclose(ordf["or"].to_numpy(), expected_or, rtol=1e-10)


def test_odds_ratios_ci(logit_data):
    """CI bounds are exp(ci_lower_beta), exp(ci_upper_beta)."""
    r = logit("y ~ x1 + x2", data=logit_data)
    ordf = odds_ratios(r)
    ci = r.confint()
    expected_lo = np.exp(ci[:, 0])
    expected_hi = np.exp(ci[:, 1])
    np.testing.assert_allclose(ordf["ci_lower"].to_numpy(), expected_lo, rtol=1e-10)
    np.testing.assert_allclose(ordf["ci_upper"].to_numpy(), expected_hi, rtol=1e-10)


def test_odds_ratios_wrong_model(binary_data):
    """odds_ratios on probit raises error."""
    r = probit("y ~ x1 + x2", data=binary_data)
    with pytest.raises(ValueError, match="Logit"):
        odds_ratios(r)


# ── Additional robustness tests ───────────────────────────────────


def test_logit_nan_dropped():
    """NaN in x columns handled by dropping those rows."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    xb = 0.5 + 1.0 * x1 - 0.5 * x2
    prob = 1.0 / (1.0 + np.exp(-xb))
    y = (rng.uniform(size=n) < prob).astype(float)
    # Inject NaN
    x1[0] = np.nan
    x1[10] = np.nan
    x2[5] = np.nan
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    r = logit("y ~ x1 + x2", data=df)
    assert r.n_obs == n - 3
    assert np.all(np.isfinite(r.coefficients))


def test_logit_null_dropped():
    """Polars nulls in x columns handled by dropping those rows."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    xb = 0.5 + 1.0 * x1 - 0.5 * x2
    prob = 1.0 / (1.0 + np.exp(-xb))
    y = (rng.uniform(size=n) < prob).astype(float)
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})
    # Inject Polars nulls
    null_mask = pl.Series("m", [False] * n)
    null_mask[0] = True
    null_mask[10] = True
    null_mask[5] = True
    df = df.with_columns(pl.when(null_mask).then(None).otherwise(pl.col("x1")).alias("x1"))
    assert df["x1"].null_count() == 3

    r = logit("y ~ x1 + x2", data=df)
    assert r.n_obs == n - 3
    assert np.all(np.isfinite(r.coefficients))


def test_logit_inf_dropped():
    """Inf in x columns handled by dropping those rows."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    xb = 0.5 + 1.0 * x1 - 0.5 * x2
    prob = 1.0 / (1.0 + np.exp(-xb))
    y = (rng.uniform(size=n) < prob).astype(float)
    # Inject inf
    x1[0] = np.inf
    x1[10] = -np.inf
    x2[5] = np.inf
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2})

    r = logit("y ~ x1 + x2", data=df)
    assert r.n_obs == n - 3
    assert np.all(np.isfinite(r.coefficients))


def test_probit_lazyframe():
    """LazyFrame input works for probit."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    xb = 0.5 + 1.0 * x1
    prob = stats.norm.cdf(xb)
    y = (rng.uniform(size=n) < prob).astype(float)
    df = pl.DataFrame({"y": y, "x1": x1}).lazy()
    r = probit("y ~ x1", data=df)
    assert r.model_type == "Probit"
    assert r.n_obs == n


def test_marginal_effects_finite(binary_data):
    """marginal_effects() returns all finite values."""
    r = probit("y ~ x1 + x2", data=binary_data)
    me = marginal_effects(r, at="mean")
    assert np.all(np.isfinite(me["dy_dx"].to_numpy()))
    assert np.all(np.isfinite(me["se"].to_numpy()))


def test_odds_ratios_positive(logit_data):
    """odds_ratios() returns all positive values."""
    r = logit("y ~ x1 + x2", data=logit_data)
    ordf = odds_ratios(r)
    assert np.all(ordf["or"].to_numpy() > 0)
    assert np.all(ordf["se"].to_numpy() > 0)
    assert np.all(ordf["ci_lower"].to_numpy() > 0)
    assert np.all(ordf["ci_upper"].to_numpy() > 0)
