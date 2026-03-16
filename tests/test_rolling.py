"""Tests for rolling-window regression."""

import numpy as np
import polars as pl
import pytest

import polars_reg as pr


@pytest.fixture
def rolling_data():
    """Time-series dataset for rolling regression."""
    rng = np.random.default_rng(42)
    n = 100
    t = np.arange(n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    e = rng.standard_normal(n) * 0.5
    y = 1.0 * x1 - 0.5 * x2 + e
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "t": t})


@pytest.fixture
def panel_data():
    """Panel dataset with entities and time for grouped rolling."""
    rng = np.random.default_rng(42)
    entities = ["A", "B", "C"]
    n_per = 50
    dfs = []
    for ent in entities:
        t = np.arange(n_per)
        x1 = rng.standard_normal(n_per)
        e = rng.standard_normal(n_per) * 0.5
        y = 2.0 * x1 + e
        dfs.append(pl.DataFrame({"y": y, "x1": x1, "entity": [ent] * n_per, "t": t}))
    return pl.concat(dfs)


def test_rolling_basic(rolling_data):
    """Basic rolling OLS: verify number of results matches expected window count."""
    window = 20
    result = pr.rolling_reg(pr.ols, "y ~ x1 + x2", rolling_data, time="t", window=window)
    # With 100 periods and window=20, stride=1: 100 - 20 + 1 = 81 windows
    expected = 100 - window + 1
    assert len(result) == expected
    assert len(result.failed) == 0


def test_rolling_stride(rolling_data):
    """Stride > 1: verify result count and key spacing."""
    window = 20
    stride = 5
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=window,
        stride=stride,
    )
    n_periods = 100
    expected = len(range(0, n_periods - window + 1, stride))
    assert len(result) == expected

    # Keys should be spaced by stride
    keys = list(result.keys())
    for i in range(1, len(keys)):
        assert keys[i] - keys[i - 1] == stride


def test_rolling_group_by(panel_data):
    """Rolling with group_by: verify tuple keys and per-entity results."""
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1",
        panel_data,
        time="t",
        window=20,
        group_by="entity",
    )
    # 3 entities * (50 - 20 + 1) = 3 * 31 = 93 windows
    assert len(result) == 93

    # Keys should be tuples
    for key in result.keys():
        assert isinstance(key, tuple)
        assert len(key) == 2
        entity, t_end = key
        assert entity in ("A", "B", "C")


def test_rolling_min_obs(rolling_data):
    """Windows below min_obs should be skipped."""
    # Create sparse data with some periods having very few obs
    # Use window_type="obs" with min_obs > window to skip all
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
        min_obs=200,
    )
    assert len(result) == 0
    assert len(result.failed) > 0


def test_rolling_window_exceeds_periods(rolling_data):
    """Window > number of periods: empty result, no error."""
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=200,
    )
    assert len(result) == 0
    assert len(result.failed) == 0


def test_rolling_coef_series(rolling_data):
    """Verify .coef_series() output shape and columns."""
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
        stride=10,
    )
    cs = result.coef_series()
    assert "time" in cs.columns
    assert "variable" in cs.columns
    assert "coefficient" in cs.columns
    assert "se" in cs.columns
    assert "ci_lower" in cs.columns
    assert "ci_upper" in cs.columns
    # Each window has 3 coefficients (x1, x2, _cons)
    n_windows = len(result)
    assert len(cs) == n_windows * 3


def test_rolling_coef_table(rolling_data):
    """Verify .coef_table() has expected columns."""
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
        stride=10,
    )
    table = result.coef_table()
    assert "window" in table.columns
    assert "name" in table.columns
    assert "coef" in table.columns
    assert "se" in table.columns
    n_windows = len(result)
    assert len(table) == n_windows * 3


def test_rolling_plot_coefs(rolling_data):
    """Verify .plot_coefs() returns an object without error."""
    pytest.importorskip("altair")
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
        stride=10,
    )
    chart = result.plot_coefs()
    assert chart is not None


def test_rolling_summary(rolling_data):
    """Verify .summary() returns a string with expected content."""
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
    )
    s = result.summary()
    assert isinstance(s, str)
    assert "Rolling Regression" in s
    assert "window=20" in s
    assert "windows succeeded" in s


def test_rolling_kwargs_passthrough(rolling_data):
    """Verify vcov/cluster kwargs are passed through to the estimator."""
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
        stride=20,
        vcov="HC1",
    )
    for r in result.values():
        assert r.vcov_type == "HC1"


def test_rolling_failed_windows():
    """Windows with singular matrices should end up in .failed."""
    # Create data where some windows are singular (constant x)
    rng = np.random.default_rng(42)
    n = 40
    x1 = np.concatenate(
        [
            np.ones(10),  # first 10 periods: no variation -> singular
            rng.standard_normal(30),
        ]
    )
    y = rng.standard_normal(n)
    df = pl.DataFrame({"y": y, "x1": x1, "t": np.arange(n)})

    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1",
        df,
        time="t",
        window=10,
        stride=10,
    )
    # First window (t=0..9) has constant x1 -> should fail
    assert len(result.failed) > 0


def test_rolling_lazyframe(rolling_data):
    """Verify LazyFrame input works."""
    lazy = rolling_data.lazy()
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        lazy,
        time="t",
        window=20,
        stride=20,
    )
    assert len(result) > 0
    for r in result.values():
        assert isinstance(r, pr.RegressionResult)


def test_rolling_seam_fe():
    """Rolling with FE absorption produces finite results."""
    rng = np.random.default_rng(42)
    n = 200
    t = np.repeat(np.arange(50), 4)
    fe = np.tile(["a", "b", "c", "d"], 50)
    x1 = rng.standard_normal(n)
    y = x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "t": t, "fe": fe})

    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 | fe",
        df,
        time="t",
        window=20,
        stride=10,
    )
    assert len(result) > 0
    for r in result.values():
        assert np.all(np.isfinite(r.coefficients))


def test_rolling_monotonic_keys(rolling_data):
    """Result keys should be in chronological order."""
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
    )
    keys = list(result.keys())
    for i in range(1, len(keys)):
        assert keys[i] > keys[i - 1]


def test_rolling_store_residuals(rolling_data):
    """store_residuals=False strips residuals; True keeps them."""
    result_no = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
        stride=20,
        store_residuals=False,
    )
    result_yes = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
        stride=20,
        store_residuals=True,
    )
    for r in result_no.values():
        assert len(r.residuals) == 0
        assert r._X is None
        assert r._y is None

    for r in result_yes.values():
        assert len(r.residuals) > 0


def test_rolling_by_entity(panel_data):
    """Verify .by_entity() works with group_by results."""
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1",
        panel_data,
        time="t",
        window=20,
        group_by="entity",
    )
    by_ent = result.by_entity()
    assert "A" in by_ent
    assert "B" in by_ent
    assert "C" in by_ent
    # Each entity should have (50 - 20 + 1) = 31 windows
    for ent_result in by_ent.values():
        assert isinstance(ent_result, pr.RollingRegressionResult)
        assert len(ent_result) == 31
        # Keys should be scalar (window end), not tuples
        for key in ent_result.keys():
            assert not isinstance(key, tuple)


def test_rolling_by_entity_no_groups_raises(rolling_data):
    """Verify .by_entity() raises ValueError without group_by."""
    result = pr.rolling_reg(
        pr.ols,
        "y ~ x1 + x2",
        rolling_data,
        time="t",
        window=20,
    )
    with pytest.raises(ValueError, match="tuple keys"):
        result.by_entity()


def test_rolling_window_type_obs():
    """Verify window_type='obs' counts rows not periods."""
    rng = np.random.default_rng(42)

    # ── Part 1: obs-mode with 1 obs per period (no key collisions) ──
    n = 30
    t = np.arange(n)
    x1 = rng.standard_normal(n)
    y = x1 + rng.standard_normal(n) * 0.5
    df_simple = pl.DataFrame({"y": y, "x1": x1, "t": t})

    result_obs = pr.rolling_reg(
        pr.ols,
        "y ~ x1",
        df_simple,
        time="t",
        window=6,
        window_type="obs",
    )
    # 30 - 6 + 1 = 25 windows (1 obs per period -> no key collision)
    assert len(result_obs) == 25

    # ── Part 2: periods-mode with multiple obs per period ────────────
    n_per_t = 3
    n_t = 10
    n2 = n_per_t * n_t
    t2 = np.repeat(np.arange(n_t), n_per_t)
    x1_2 = rng.standard_normal(n2)
    y2 = x1_2 + rng.standard_normal(n2) * 0.5
    df_multi = pl.DataFrame({"y": y2, "x1": x1_2, "t": t2})

    result_periods = pr.rolling_reg(
        pr.ols,
        "y ~ x1",
        df_multi,
        time="t",
        window=6,
        window_type="periods",
    )
    # 10 - 6 + 1 = 5 windows (each with 18 obs)
    assert len(result_periods) == 5

    # ── Part 3: obs vs periods give different counts ─────────────────
    # Same multi-obs data: obs-mode gives more windows (sliding by rows)
    result_obs_multi = pr.rolling_reg(
        pr.ols,
        "y ~ x1",
        df_multi,
        time="t",
        window=6,
        window_type="obs",
    )
    # obs-mode: 30 - 6 + 1 = 25 potential windows, but many share
    # the same end-time key, so fewer unique keys are stored.
    assert len(result_obs_multi) < 25
    assert len(result_obs_multi) != len(result_periods)


# ── Data quality edge cases ──────────────────────────────────────


def test_rolling_with_nulls_in_data():
    """Polars nulls in y should be transparently dropped per window."""
    rng = np.random.default_rng(42)
    n = 100
    t = np.arange(n)
    x = rng.standard_normal(n)
    y = 1.5 * x + rng.standard_normal(n) * 0.5

    df = pl.DataFrame({"y": y, "x": x, "t": t})

    # Inject Polars nulls into ~10 rows of y
    null_mask = pl.Series("m", [i in {5, 15, 25, 35, 45, 55, 65, 75, 85, 95} for i in range(n)])
    df = df.with_columns(pl.when(null_mask).then(None).otherwise(pl.col("y")).alias("y"))
    assert df["y"].null_count() == 10

    result = pr.rolling_reg(pr.ols, "y ~ x", df, time="t", window=20)

    assert len(result) > 0
    for r in result.values():
        assert np.all(np.isfinite(r.coefficients))
        assert np.all(np.isfinite(r.se))


def test_rolling_with_nan_in_data():
    """NaN values in x should be converted to null and dropped by extract_arrays."""
    rng = np.random.default_rng(42)
    n = 100
    t = np.arange(n)
    x = rng.standard_normal(n)
    y = 1.5 * x + rng.standard_normal(n) * 0.5

    # Inject float NaN into x via numpy before converting to Polars
    x_with_nan = x.copy()
    nan_indices = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]
    x_with_nan[nan_indices] = float("nan")

    df = pl.DataFrame({"y": y, "x": x_with_nan, "t": t})

    result = pr.rolling_reg(pr.ols, "y ~ x", df, time="t", window=20)

    assert len(result) > 0
    for r in result.values():
        assert np.all(np.isfinite(r.coefficients))
        assert np.all(np.isfinite(r.se))


def test_rolling_with_inf_in_data():
    """Inf/-inf values should be converted to null and dropped by extract_arrays."""
    rng = np.random.default_rng(42)
    n = 100
    t = np.arange(n)
    x = rng.standard_normal(n)
    y = 1.5 * x + rng.standard_normal(n) * 0.5

    # Inject inf and -inf into a few rows
    x_with_inf = x.copy()
    x_with_inf[10] = np.inf
    x_with_inf[30] = -np.inf
    x_with_inf[50] = np.inf

    df = pl.DataFrame({"y": y, "x": x_with_inf, "t": t})

    result = pr.rolling_reg(pr.ols, "y ~ x", df, time="t", window=20)

    assert len(result) > 0
    for r in result.values():
        assert np.all(np.isfinite(r.coefficients))
        assert np.all(np.isfinite(r.se))


def test_rolling_with_time_gaps():
    """Non-consecutive time values: windows should span distinct time values."""
    rng = np.random.default_rng(42)
    # Periods [0,1,2,3,4,10,11,12,13,14] — gap at 5-9
    time_vals = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    n = len(time_vals)
    x = rng.standard_normal(n)
    y = 2.0 * x + rng.standard_normal(n) * 0.3

    df = pl.DataFrame({"y": y, "x": x, "t": time_vals})

    result = pr.rolling_reg(pr.ols, "y ~ x", df, time="t", window=5, window_type="periods")

    # 10 distinct time values, window=5 => 10 - 5 + 1 = 6 windows
    assert len(result) == 6

    # Keys should be actual time period values (not indices)
    keys = list(result.keys())
    # The window ends at positions [4, 5, 6, 7, 8, 9] in the sorted unique times
    expected_keys = [time_vals[i] for i in range(4, 10)]
    assert keys == expected_keys

    # Verify all results are finite
    for r in result.values():
        assert np.all(np.isfinite(r.coefficients))


def test_rolling_unbalanced_panel():
    """Panel where entities have different numbers of time periods."""
    rng = np.random.default_rng(42)

    # Entity A: 50 time periods
    t_a = np.arange(50)
    x_a = rng.standard_normal(50)
    y_a = 1.0 * x_a + rng.standard_normal(50) * 0.5
    df_a = pl.DataFrame(
        {
            "y": y_a,
            "x": x_a,
            "entity": ["A"] * 50,
            "t": t_a,
        }
    )

    # Entity B: 30 time periods
    t_b = np.arange(30)
    x_b = rng.standard_normal(30)
    y_b = 2.0 * x_b + rng.standard_normal(30) * 0.5
    df_b = pl.DataFrame(
        {
            "y": y_b,
            "x": x_b,
            "entity": ["B"] * 30,
            "t": t_b,
        }
    )

    df = pl.concat([df_a, df_b])

    result = pr.rolling_reg(
        pr.ols,
        "y ~ x",
        df,
        time="t",
        window=20,
        group_by="entity",
    )

    by_ent = result.by_entity()
    assert "A" in by_ent
    assert "B" in by_ent

    # A has more windows than B since it has more periods
    # A: uses global unique times (50 periods), windows where entity has data
    # B: only has data for 30 periods, windows beyond that will fail or be empty
    assert len(by_ent["A"]) > 0
    assert len(by_ent["B"]) > 0
    assert len(by_ent["A"]) > len(by_ent["B"])

    # All successful results should have finite coefficients
    for ent_result in by_ent.values():
        for r in ent_result.values():
            assert np.all(np.isfinite(r.coefficients))
