import numpy as np
import pytest

alt = pytest.importorskip("altair")

from polars_reg._plotting import _partial_residuals, avplot, coefplot  # noqa: E402
from polars_reg._results import RegressionResult  # noqa: E402


def _make_result(names=None, store_Xy=True):
    """Build a simple RegressionResult for testing."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    ones = np.ones(n)
    X = np.column_stack([x1, x2, ones])
    beta_true = np.array([1.5, -0.8, 3.0])
    y = X @ beta_true + rng.standard_normal(n) * 0.5

    # OLS fit
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    k = X.shape[1]
    s2 = float(resid @ resid / (n - k))
    vcov = s2 * np.linalg.inv(X.T @ X)
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)

    if names is None:
        names = ["x1", "x2", "_cons"]

    result = RegressionResult(
        coefficients=beta_hat,
        vcov=vcov,
        residuals=resid,
        names=names,
        n_obs=n,
        k=k,
        df_r=n - k,
        r_squared=r2,
        r_squared_adj=1 - (1 - r2) * (n - 1) / (n - k - 1),
        model_type="OLS",
        vcov_type="iid",
        _X=X if store_Xy else None,
        _y=y if store_Xy else None,
    )
    return result


# --- coefplot tests ---


def test_coefplot_returns_chart():
    res = _make_result()
    chart = coefplot(res)
    assert isinstance(chart, (alt.Chart, alt.LayerChart))


def test_coefplot_excludes_cons():
    res = _make_result()
    chart = coefplot(res)
    # Check the underlying data doesn't contain _cons
    chart_dict = chart.to_dict()
    # Find data in the spec — look for variable values in datasets
    found_cons = False
    for dataset in chart_dict.get("datasets", {}).values():
        for row in dataset:
            if row.get("variable") == "_cons":
                found_cons = True
    assert not found_cons


def test_coefplot_variables_filter():
    res = _make_result()
    chart = coefplot(res, variables=["x1"])
    chart_dict = chart.to_dict()
    variables_found = set()
    for dataset in chart_dict.get("datasets", {}).values():
        for row in dataset:
            if "variable" in row:
                variables_found.add(row["variable"])
    assert "x1" in variables_found
    assert "x2" not in variables_found


def test_coefplot_exclude_filter():
    res = _make_result()
    chart = coefplot(res, exclude=["_cons", "x2"])
    chart_dict = chart.to_dict()
    variables_found = set()
    for dataset in chart_dict.get("datasets", {}).values():
        for row in dataset:
            if "variable" in row:
                variables_found.add(row["variable"])
    assert "x1" in variables_found
    assert "x2" not in variables_found
    assert "_cons" not in variables_found


def test_coefplot_multi_model_with_labels():
    res1 = _make_result()
    res2 = _make_result()
    chart = coefplot(res1, res2, labels=["Model A", "Model B"])
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
    chart_dict = chart.to_dict()
    models_found = set()
    for dataset in chart_dict.get("datasets", {}).values():
        for row in dataset:
            if "model" in row:
                models_found.add(row["model"])
    assert "Model A" in models_found
    assert "Model B" in models_found


# --- avplot tests ---


def test_avplot_single_variable_returns_chart():
    res = _make_result()
    chart = avplot(res, variable="x1")
    assert isinstance(chart, (alt.Chart, alt.LayerChart))


def test_avplot_all_variables_returns_concat():
    res = _make_result()
    chart = avplot(res)
    assert isinstance(chart, (alt.VConcatChart, alt.HConcatChart, alt.ConcatChart))


def test_avplot_slope_equals_coefficient():
    """FWL theorem: regressing e_y on e_x should give the same slope as
    the full regression coefficient for that variable."""
    res = _make_result()
    X = res._X
    y = res._y
    names = res.names

    for j, name in enumerate(names):
        if name == "_cons":
            continue
        e_y, e_x = _partial_residuals(X, y, j)

        # Regress e_y on e_x (simple OLS, no intercept)
        slope_fwl = float(e_x @ e_y / (e_x @ e_x))
        np.testing.assert_allclose(
            slope_fwl,
            res.coefficients[j],
            atol=1e-10,
            err_msg=f"FWL slope mismatch for {name}",
        )


def test_avplot_raises_without_Xy():
    res = _make_result(store_Xy=False)
    with pytest.raises(ValueError, match="_X and _y"):
        avplot(res, variable="x1")


def test_avplot_skips_cons_in_grid():
    res = _make_result()
    chart = avplot(res)
    chart_dict = chart.to_dict()
    # Check no subplot has _cons in its title
    titles = []

    def _collect_titles(d):
        if isinstance(d, dict):
            if "title" in d and isinstance(d["title"], str):
                titles.append(d["title"])
            for v in d.values():
                _collect_titles(v)
        elif isinstance(d, list):
            for item in d:
                _collect_titles(item)

    _collect_titles(chart_dict)
    assert not any("_cons" in t for t in titles)


def test_coefplot_method_on_result():
    """RegressionResult.coefplot() convenience method works."""
    res = _make_result()
    chart = res.coefplot()
    assert isinstance(chart, (alt.Chart, alt.LayerChart))


def test_avplot_method_on_result():
    """RegressionResult.avplot() convenience method works."""
    res = _make_result()
    chart = res.avplot(variable="x1")
    assert isinstance(chart, (alt.Chart, alt.LayerChart))
