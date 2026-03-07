"""Tests for bootstrap standard errors."""

import numpy as np
import polars as pl
import pytest

from polars_reg import ols


@pytest.fixture
def boot_data():
    """Data for bootstrap tests."""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    e = rng.standard_normal(n) * 0.5
    y = 2.0 + 1.5 * x1 - 0.5 * x2 + e
    firm = np.repeat(np.arange(50), 10)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm})


def test_pairs_bootstrap_runs(boot_data):
    """Pairs bootstrap produces valid VCV."""
    r = ols("y ~ x1 + x2", data=boot_data, vcov="bootstrap", n_boot=199, seed=42)
    assert r.vcov_type == "bootstrap"
    assert np.all(r.se > 0)
    assert r.se.shape == (3,)  # x1, x2, _cons


def test_pairs_bootstrap_close_to_robust(boot_data):
    """Bootstrap SEs should be in the same ballpark as robust SEs."""
    r_hc1 = ols("y ~ x1 + x2", data=boot_data, vcov="HC1")
    r_boot = ols("y ~ x1 + x2", data=boot_data, vcov="bootstrap", n_boot=999, seed=42)
    # Bootstrap SEs should be within 50% of HC1 SEs (loose check for randomness)
    ratio = r_boot.se / r_hc1.se
    assert np.all(ratio > 0.5)
    assert np.all(ratio < 2.0)


def test_pairs_bootstrap_reproducible(boot_data):
    """Same seed gives same results."""
    r1 = ols("y ~ x1 + x2", data=boot_data, vcov="bootstrap", n_boot=199, seed=123)
    r2 = ols("y ~ x1 + x2", data=boot_data, vcov="bootstrap", n_boot=199, seed=123)
    np.testing.assert_allclose(r1.se, r2.se)


def test_pairs_bootstrap_different_seeds(boot_data):
    """Different seeds give different results."""
    r1 = ols("y ~ x1 + x2", data=boot_data, vcov="bootstrap", n_boot=199, seed=1)
    r2 = ols("y ~ x1 + x2", data=boot_data, vcov="bootstrap", n_boot=199, seed=2)
    assert not np.allclose(r1.se, r2.se)


def test_wild_bootstrap_runs(boot_data):
    """Wild cluster bootstrap produces valid VCV."""
    r = ols(
        "y ~ x1 + x2",
        data=boot_data,
        vcov="wildboot",
        cluster="firm",
        n_boot=199,
        seed=42,
    )
    assert r.vcov_type == "wildboot"
    assert np.all(r.se > 0)


def test_wild_bootstrap_requires_cluster(boot_data):
    """Wild bootstrap without cluster raises ValueError."""
    with pytest.raises(ValueError, match="requires cluster"):
        ols("y ~ x1 + x2", data=boot_data, vcov="wildboot", n_boot=99, seed=42)


def test_wild_bootstrap_close_to_cluster(boot_data):
    """Wild bootstrap SEs should be in the same ballpark as cluster-robust SEs."""
    r_cl = ols("y ~ x1 + x2", data=boot_data, cluster="firm")
    r_wb = ols(
        "y ~ x1 + x2",
        data=boot_data,
        vcov="wildboot",
        cluster="firm",
        n_boot=999,
        seed=42,
    )
    ratio = r_wb.se / r_cl.se
    assert np.all(ratio > 0.3)
    assert np.all(ratio < 3.0)


def test_wild_bootstrap_reproducible(boot_data):
    """Same seed gives same results."""
    r1 = ols(
        "y ~ x1 + x2",
        data=boot_data,
        vcov="wildboot",
        cluster="firm",
        n_boot=199,
        seed=42,
    )
    r2 = ols(
        "y ~ x1 + x2",
        data=boot_data,
        vcov="wildboot",
        cluster="firm",
        n_boot=199,
        seed=42,
    )
    np.testing.assert_allclose(r1.se, r2.se)


def test_bootstrap_with_fe(boot_data):
    """Pairs bootstrap works with absorbed FE."""
    r = ols(
        "y ~ x1 + x2 | firm",
        data=boot_data,
        vcov="bootstrap",
        n_boot=199,
        seed=42,
    )
    assert r.fe_absorbed == ["firm"]
    assert np.all(r.se > 0)


def test_bootstrap_coefficients_unchanged(boot_data):
    """Bootstrap should not change point estimates, only SEs."""
    r_iid = ols("y ~ x1 + x2", data=boot_data)
    r_boot = ols("y ~ x1 + x2", data=boot_data, vcov="bootstrap", n_boot=199, seed=42)
    np.testing.assert_allclose(r_boot.coefficients, r_iid.coefficients, atol=1e-10)


def test_bootstrap_summary(boot_data):
    """Bootstrap result summary displays correctly."""
    r = ols("y ~ x1 + x2", data=boot_data, vcov="bootstrap", n_boot=99, seed=42)
    s = r.summary()
    assert "bootstrap" in s
    assert "x1" in s
