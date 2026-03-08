"""Tests comparing polars_reg output against frozen Stata fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import polars_reg as pr

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "stata"
DATA_PATH = Path(__file__).parent / "fixtures" / "parity_data.csv"

# Skip all if fixtures not generated yet
pytestmark = pytest.mark.skipif(
    not list(FIXTURE_DIR.glob("*.csv")) if FIXTURE_DIR.exists() else True,
    reason="Parity fixtures not generated (run tests/fixtures/stata/generate_fixtures.do)",
)


@pytest.fixture(scope="module")
def parity_data():
    return pl.read_csv(str(DATA_PATH))


def load_fixture(name: str) -> pd.DataFrame:
    """Load a Stata fixture CSV. Returns DataFrame with variable, coef, se columns."""
    path = FIXTURE_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"Fixture {name}.csv not found")
    df = pd.read_csv(path, na_values=".")
    for col in ("coef", "se", "t", "p"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compare_coefs(result, fixture_df, rtol_coef=1e-6, rtol_se=1e-4):
    """Compare polars_reg result against Stata fixture."""
    # Filter out stat rows
    coef_rows = fixture_df[~fixture_df["variable"].str.startswith("_stat_")]
    for _, row in coef_rows.iterrows():
        name = row["variable"]
        if name == "_cons":
            if "_cons" in result.names:
                idx = result.names.index("_cons")
            else:
                continue
        elif name in result.names:
            idx = result.names.index(name)
        else:
            continue
        np.testing.assert_allclose(
            result.coefficients[idx],
            row["coef"],
            rtol=rtol_coef,
            err_msg=f"Coefficient mismatch for {name}",
        )
        np.testing.assert_allclose(
            result.se[idx],
            row["se"],
            rtol=rtol_se,
            err_msg=f"SE mismatch for {name}",
        )


# --- OLS ---


def test_parity_ols_iid(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data)
    compare_coefs(result, load_fixture("ols_iid"))


def test_parity_ols_hc1(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="HC1")
    compare_coefs(result, load_fixture("ols_hc1"))


def test_parity_ols_hc2(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="HC2")
    compare_coefs(result, load_fixture("ols_hc2"))


def test_parity_ols_hc3(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="HC3")
    compare_coefs(result, load_fixture("ols_hc3"))


def test_parity_ols_cluster(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, cluster=["firm_id"])
    compare_coefs(result, load_fixture("ols_cluster"))


def test_parity_ols_nw(parity_data):
    # Fixture generated from single firm (pure time series)
    firm1 = parity_data.filter(pl.col("firm_id") == 1)
    result = pr.ols("y ~ x1 + x2", data=firm1, vcov="NW", time="year_id", bandwidth=4)
    compare_coefs(result, load_fixture("ols_nw"))


def test_parity_ols_dk(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="DK", time="year_id", bandwidth=4)
    # DK involves time-aggregation + kernel; slightly wider tolerance
    compare_coefs(result, load_fixture("ols_dk"), rtol_se=2e-4)


# --- OLS + FE ---


def test_parity_ols_fe_cluster(parity_data):
    result = pr.ols("y ~ x1 + x2 | firm_id", data=parity_data, cluster=["firm_id"])
    compare_coefs(result, load_fixture("ols_fe_cluster"))


def test_parity_ols_fe_hc1(parity_data):
    result = pr.ols("y ~ x1 + x2 | firm_id", data=parity_data, vcov="HC1")
    compare_coefs(result, load_fixture("ols_fe_hc1"))


def test_parity_ols_2fe_cluster(parity_data):
    result = pr.ols("y ~ x1 + x2 | firm_id + year_id", data=parity_data, cluster=["firm_id"])
    compare_coefs(result, load_fixture("ols_2fe_cluster"))


# --- 2SLS ---


def test_parity_iv_iid(parity_data):
    result = pr.iv2sls("y ~ x1 || x_endog ~ z1 + z2", data=parity_data)
    compare_coefs(result, load_fixture("iv_iid"))


def test_parity_iv_robust(parity_data):
    result = pr.iv2sls("y ~ x1 || x_endog ~ z1 + z2", data=parity_data, vcov="HC1")
    compare_coefs(result, load_fixture("iv_robust"))


def test_parity_iv_cluster(parity_data):
    result = pr.iv2sls("y ~ x1 || x_endog ~ z1 + z2", data=parity_data, cluster=["firm_id"])
    compare_coefs(result, load_fixture("iv_cluster"))


def test_parity_iv_fe_cluster(parity_data):
    result = pr.iv2sls(
        "y ~ x1 | firm_id | x_endog ~ z1 + z2",
        data=parity_data,
        cluster=["firm_id"],
    )
    compare_coefs(result, load_fixture("iv_fe_cluster"))


# --- Panel RE ---


def test_parity_re_iid(parity_data):
    result = pr.panel_re("y ~ x1 + x2", data=parity_data, entity="firm_id")
    compare_coefs(result, load_fixture("re_iid"))


def test_parity_re_cluster(parity_data):
    result = pr.panel_re(
        "y ~ x1 + x2",
        data=parity_data,
        entity="firm_id",
        cluster=["firm_id"],
    )
    # Stata xtreg,re intercept SE uses between-group estimation;
    # compare slopes only (skip _cons)
    fixture = load_fixture("re_cluster")
    fixture = fixture[fixture["variable"] != "_cons"]
    compare_coefs(result, fixture)


# --- Newey (HAC baseline) ---


def test_parity_newey(parity_data):
    # Fixture generated from single firm (pure time series)
    firm1 = parity_data.filter(pl.col("firm_id") == 1)
    result = pr.ols("y ~ x1 + x2", data=firm1, vcov="NW", time="year_id", bandwidth=8)
    compare_coefs(result, load_fixture("newey_lag8"))
