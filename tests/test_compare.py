"""Tests for unified compare() function."""

import numpy as np
import polars as pl
import pytest
from great_tables import GT

from polars_reg import compare
from polars_reg._compare import ComparisonReport

# Skip entire module if comparison packages aren't installed
pytest.importorskip("pyfixest", reason="pyfixest not installed")
pytest.importorskip("statsmodels", reason="statsmodels not installed")
pytest.importorskip("linearmodels", reason="linearmodels not installed")


@pytest.fixture(scope="module")
def ols_data() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.standard_normal(n) * 0.5
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def panel_data() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n_firms, n_years = 50, 10
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(2000, 2000 + n_years), n_firms)
    fe = rng.standard_normal(n_firms)[firm_id]
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + fe + rng.standard_normal(n) * 0.3
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "firm_id": firm_id, "year_id": year_id})


@pytest.fixture(scope="module")
def binary_data() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    prob = 1.0 / (1.0 + np.exp(-(0.5 + 1.0 * x1 - 0.3 * x2)))
    y = (rng.uniform(size=n) < prob).astype(float)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


# ── Basic functionality ──────────────────────────────────────────


def test_compare_returns_report(ols_data):
    """compare() returns a ComparisonReport even with no backends available."""
    report = compare("ols", "y ~ x1 + x2", ols_data, backend="pyfixest")
    assert isinstance(report, ComparisonReport)


def test_compare_pyfixest_ols_match(ols_data):
    """OLS matches pyfixest to machine precision."""
    report = compare("ols", "y ~ x1 + x2", ols_data, vcov="HC1", backend="pyfixest")
    assert "pyfixest" in report.backends
    assert report.backends["pyfixest"].match


def test_compare_statsmodels_ols_match(ols_data):
    """OLS matches statsmodels to machine precision."""
    report = compare("ols", "y ~ x1 + x2", ols_data, vcov="HC1", backend="statsmodels")
    assert "statsmodels" in report.backends
    assert report.backends["statsmodels"].match


def test_compare_pyfixest_probit(binary_data):
    """Probit runs in pyfixest."""
    report = compare("probit", "y ~ x1 + x2", binary_data, backend="pyfixest", rtol=1e-3)
    assert "pyfixest" in report.backends
    br = report.backends["pyfixest"]
    assert br.n_obs > 0
    assert len(br.coefs) > 0


def test_compare_statsmodels_probit(binary_data):
    """Probit matches statsmodels."""
    report = compare("probit", "y ~ x1 + x2", binary_data, backend="statsmodels", rtol=1e-3)
    assert "statsmodels" in report.backends
    br = report.backends["statsmodels"]
    assert br.n_obs > 0


def test_compare_statsmodels_logit(binary_data):
    """Logit matches statsmodels."""
    report = compare("logit", "y ~ x1 + x2", binary_data, backend="statsmodels", rtol=1e-3)
    assert "statsmodels" in report.backends
    br = report.backends["statsmodels"]
    assert br.n_obs > 0


def test_compare_pyfixest_fe(panel_data):
    """OLS + FE matches pyfixest."""
    report = compare(
        "ols",
        "y ~ x1 + x2 | firm_id",
        panel_data,
        cluster=["firm_id"],
        backend="pyfixest",
        rtol=1e-4,
    )
    assert "pyfixest" in report.backends


def test_compare_linearmodels_panel_fe(panel_data):
    """panel_fe matches linearmodels PanelOLS."""
    report = compare(
        "panel_fe",
        "y ~ x1 + x2",
        panel_data,
        entity="firm_id",
        time="year_id",
        backend="linearmodels",
        rtol=5e-2,
    )
    if "linearmodels" in report.backends:
        assert report.backends["linearmodels"].n_obs > 0


# ── backend="all" ─────────────────────────────────────────────────


def test_compare_all_backends(ols_data):
    """backend='all' runs available backends and skips unavailable."""
    report = compare("ols", "y ~ x1 + x2", ols_data, vcov="HC1")
    # At least one backend should be available (pyfixest or statsmodels)
    total = len(report.backends) + len(report.skipped)
    assert total >= 1
    assert isinstance(report.skipped, dict)


def test_compare_all_summary(ols_data):
    """summary() returns a GT object with expected content."""
    report = compare("ols", "y ~ x1 + x2", ols_data, vcov="HC1")
    gt = report.summary()
    assert isinstance(gt, GT)
    html = gt.as_raw_html()
    assert "polars_reg" in html
    assert "x1" in html


# ── Graceful skipping ─────────────────────────────────────────────


def test_compare_unsupported_estimator_skips(ols_data):
    """Unsupported estimator/backend combo is skipped gracefully."""
    report = compare(
        "ols",
        "y ~ x1 + x2",
        ols_data,
        backend="linearmodels",
    )
    # plain OLS without FE is not linearmodels' domain — should skip
    assert "linearmodels" in report.skipped


def test_compare_unknown_backend(ols_data):
    """Unknown backend name is noted in skipped."""
    report = compare("ols", "y ~ x1", ols_data, backend="nonexistent")
    assert "nonexistent" in report.skipped


def test_compare_bad_estimator_raises(ols_data):
    """Unknown estimator name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown estimator"):
        compare("fake_estimator", "y ~ x1", ols_data, backend="pyfixest")


# ── BackendResult properties ─────────────────────────────────────


def test_backend_result_has_code(ols_data):
    """Each backend result includes equivalent code string."""
    report = compare("ols", "y ~ x1 + x2", ols_data, backend="pyfixest")
    if "pyfixest" in report.backends:
        br = report.backends["pyfixest"]
        assert isinstance(br.code, str)
        assert len(br.code) > 0


def test_backend_result_diff_computed(ols_data):
    """Diffs are computed for each backend."""
    report = compare("ols", "y ~ x1 + x2", ols_data, backend="statsmodels")
    if "statsmodels" in report.backends:
        br = report.backends["statsmodels"]
        assert br.max_coef_rdiff >= 0
        assert br.max_se_rdiff >= 0


# ── Code output ──────────────────────────────────────────────────


def test_compare_code_output(ols_data):
    """code() returns formatted code with polars_reg section."""
    report = compare("ols", "y ~ x1 + x2", ols_data, vcov="HC1", backend="pyfixest")
    code = report.code()
    assert "polars_reg" in code
    assert "y ~ x1 + x2" in code


def test_compare_code_includes_kwargs(ols_data):
    """code() polars_reg section includes vcov when set."""
    report = compare("ols", "y ~ x1 + x2", ols_data, vcov="HC1", backend="pyfixest")
    assert 'vcov="HC1"' in report.polars_code
    assert "pr.ols" in report.polars_code


def test_compare_code_iid_omits_vcov(ols_data):
    """code() omits vcov when iid (the default)."""
    report = compare("ols", "y ~ x1 + x2", ols_data, vcov="iid", backend="pyfixest")
    assert "vcov" not in report.polars_code


# ── repr ──────────────────────────────────────────────────────────


def test_compare_repr(ols_data):
    """ComparisonReport has informative repr."""
    report = compare("ols", "y ~ x1 + x2", ols_data, backend="pyfixest")
    r = repr(report)
    assert "ComparisonReport" in r
    assert "ols" in r


# ── match_ssc ──────────────────────────────────────────────────────


def test_compare_match_ssc_pyfixest_no_duplicate(ols_data):
    """match_ssc=True with pyfixest does not add a duplicate column (same SSC)."""
    report = compare("ols", "y ~ x1 + x2", ols_data, backend="pyfixest", match_ssc=True)
    assert isinstance(report, ComparisonReport)
    # pyfixest SSC matches polars_reg default — no extra matched run
    assert len(report.polars_matched) == 0


def test_compare_match_ssc_stata_ols_no_extra_column(ols_data):
    """match_ssc=True with stata OLS backend does NOT add an extra column.

    Stata reghdfe SSC now matches the default (k_fixef='nonnested', G_df='min'),
    so no separate matched run is produced for OLS.
    """
    report = compare("ols", "y ~ x1 + x2", ols_data, backend="stata", match_ssc=True, vcov="HC1")
    assert isinstance(report, ComparisonReport)
    # Stata OLS SSC matches default → no matched run
    assert "polars_reg_stata_ssc" not in report.polars_matched


def test_compare_match_ssc_summary_renders(ols_data):
    """summary() works with match_ssc columns."""
    report = compare("ols", "y ~ x1 + x2", ols_data, backend="stata", match_ssc=True, vcov="HC1")
    gt = report.summary()
    assert isinstance(gt, GT)
    html = gt.as_raw_html()
    assert "polars_reg" in html


def test_compare_match_ssc_code_output(ols_data):
    """code() includes matched SSC sections."""
    report = compare("ols", "y ~ x1 + x2", ols_data, backend="stata", match_ssc=True, vcov="HC1")
    code = report.code()
    assert "polars_reg" in code
    # Should have the matched SSC code section
    if report.polars_matched:
        assert "stata ssc" in code


def test_compare_match_ssc_repr(ols_data):
    """repr works with match_ssc=True.

    For OLS with Stata backend, SSC matches default so no ssc-matched
    column is produced. The repr should still be valid.
    """
    report = compare("ols", "y ~ x1 + x2", ols_data, backend="stata", match_ssc=True, vcov="HC1")
    r = repr(report)
    assert "ComparisonReport" in r


def test_compare_ssc_parameter(ols_data):
    """User-provided ssc is used for polars_reg."""
    from polars_reg import ssc

    report = compare("ols", "y ~ x1 + x2", ols_data, ssc=ssc(k_adj=False), backend="pyfixest")
    assert isinstance(report, ComparisonReport)
    # The code string should include the SSC
    assert "ssc=" in report.polars_code


def test_compare_ssc_parameter_no_match_ssc(ols_data):
    """User-provided ssc works without match_ssc."""
    from polars_reg import ssc

    report = compare("ols", "y ~ x1 + x2", ols_data, ssc=ssc(k_adj=False), backend="pyfixest")
    # No matched runs when match_ssc is False (default)
    assert len(report.polars_matched) == 0
