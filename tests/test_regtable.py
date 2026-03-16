"""Tests for regtable (Great Tables-based side-by-side regression table)."""

import numpy as np
import polars as pl
import pytest
from great_tables import GT

from polars_reg import group_by_reg, ols, regtable
from polars_reg._regtable import _normalize_stat


def _html(table: GT) -> str:
    """Convenience: get HTML string from a GT table."""
    return table.as_raw_html()


def _latex(table: GT) -> str:
    """Convenience: get LaTeX string from a GT table."""
    return table.as_latex()


# ── Basic tests ──────────────────────────────────────────────────


def test_regtable_basic(simple_data):
    """Basic regtable with two models produces expected structure."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, r2)
    assert isinstance(table, GT)
    html = _html(table)
    assert "(1)" in html
    assert "(2)" in html
    assert "x1" in html


def test_regtable_custom_labels(simple_data):
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, r2, labels=["Base", "Full"])
    html = _html(table)
    assert "Base" in html
    assert "Full" in html


def test_regtable_label_count_mismatch(simple_data):
    r1 = ols("y ~ x1", data=simple_data)
    with pytest.raises(ValueError, match="Expected 1 labels"):
        regtable(r1, labels=["a", "b"])


def test_regtable_no_results():
    with pytest.raises(ValueError, match="At least one"):
        regtable()


def test_regtable_stars(simple_data):
    """Stars appear for significant coefficients."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    html = _html(regtable(r1, stars=True))
    assert "*" in html


def test_regtable_no_stars(simple_data):
    """No stars footnote when disabled."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    html = _html(regtable(r1, stars=False))
    assert "p&lt;0.10" not in html
    assert "p<0.10" not in html


# ── stat parameter tests ──────────────────────────────────────────


def test_regtable_default_shows_tstat(simple_data):
    """Default stat='t' shows t-statistics in parentheses."""
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1))
    assert "T-statistics in parentheses" in html


def test_regtable_stat_se(simple_data):
    """stat='se' shows standard errors in parentheses."""
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1, stat="se"))
    assert "Standard errors in parentheses" in html


def test_regtable_stat_none(simple_data):
    """stat=None shows coefficients only, no sub-stats."""
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1, stat=None))
    assert "parentheses" not in html
    assert "brackets" not in html


def test_regtable_stat_both(simple_data):
    """stat=('t', 'se') shows both t-stats and SEs."""
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1, stat=("t", "se")))
    assert "T-statistics in parentheses" in html
    assert "standard errors in brackets" in html
    assert "(" in html
    assert "[" in html


def test_regtable_stat_both_reversed(simple_data):
    """stat=('se', 't') shows SE first in parens, t in brackets."""
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1, stat=("se", "t")))
    assert "Standard errors in parentheses" in html
    assert "t-statistics in brackets" in html


def test_regtable_stat_values_correct(simple_data):
    """Verify actual t-stat and SE values appear in output."""
    r1 = ols("y ~ x1", data=simple_data)
    t_val = r1.tstat[0]
    se_val = r1.se[0]

    t_html = _html(regtable(r1, stat="t"))
    assert f"{t_val:.4g}" in t_html

    se_html = _html(regtable(r1, stat="se"))
    assert f"{se_val:.4g}" in se_html


# ── brackets parameter tests ──────────────────────────────────────


def test_regtable_brackets_square(simple_data):
    """brackets='square' uses [] as primary brackets."""
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1, brackets="square"))
    assert "T-statistics in brackets" in html
    assert "[" in html


def test_regtable_brackets_square_both(simple_data):
    """brackets='square' with both stats: primary in [], secondary in ()."""
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1, stat=("t", "se"), brackets="square"))
    assert "T-statistics in brackets" in html
    assert "standard errors in parentheses" in html


def test_regtable_brackets_invalid(simple_data):
    """Invalid brackets value raises ValueError."""
    r1 = ols("y ~ x1", data=simple_data)
    with pytest.raises(ValueError, match="brackets must be"):
        regtable(r1, brackets="curly")


# ── wide parameter tests ──────────────────────────────────────────


def test_regtable_wide(simple_data):
    """wide=True produces a valid GT table."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, r2, stat="t", wide=True)
    assert isinstance(table, GT)
    html = _html(table)
    assert "x1" in html


def test_regtable_wide_both(simple_data):
    """wide=True with both stats produces columns with both."""
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1, stat=("t", "se"), wide=True))
    assert "(" in html
    assert "[" in html


def test_regtable_wide_stat_none(simple_data):
    """wide=True with stat=None just shows coefficients."""
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1, stat=None, wide=True))
    assert "x1" in html


# ── Existing feature tests ────────────────────────────────────────


def test_regtable_missing_vars(simple_data):
    """Variables not in a model show as blank."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    html = _html(regtable(r1, r2))
    assert "x2" in html


def test_regtable_fe_indicator_rows(panel_data):
    """FE gets Y/N indicator rows under a 'Fixed Effects' header."""
    r1 = ols("y ~ x1 + x2", data=panel_data)
    r2 = ols("y ~ x1 + x2 | firm_id", data=panel_data)
    r3 = ols("y ~ x1 + x2 | firm_id + year_id", data=panel_data)
    html = _html(regtable(r1, r2, r3))
    assert "Fixed Effects" in html
    assert "firm_id" in html
    assert "year_id" in html


def test_regtable_cluster_indicator_rows(panel_data):
    """Cluster variable gets Y/N indicator rows."""
    r1 = ols("y ~ x1 + x2", data=panel_data)
    r2 = ols("y ~ x1 + x2 | firm_id", data=panel_data, cluster=["firm_id"])
    html = _html(regtable(r1, r2))
    assert "Clustering" in html
    assert "firm_id" in html


def test_regtable_adj_r2(simple_data):
    """Adj. R² row is present."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    html = _html(regtable(r1))
    assert "Adj." in html


def test_regtable_precision(simple_data):
    """Different precision values produce output."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    html2 = _html(regtable(r1, precision=2))
    html6 = _html(regtable(r1, precision=6))
    assert len(html2) > 0
    assert len(html6) > 0


def test_regtable_group_by_expansion():
    """GroupRegressionResult is auto-expanded with group keys as labels."""
    rng = np.random.default_rng(42)
    n = 300
    group = np.repeat(["A", "B", "C"], n // 3)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "group": group})
    grp = group_by_reg(ols, "y ~ x1", df, group_by="group")
    html = _html(regtable(grp))
    assert "A" in html
    assert "B" in html
    assert "C" in html
    assert "x1" in html


def test_regtable_group_by_mixed(simple_data):
    """Mix of RegressionResult and GroupRegressionResult works."""
    rng = np.random.default_rng(42)
    n = 200
    group = np.repeat(["X", "Y"], n // 2)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "group": group})
    grp = group_by_reg(ols, "y ~ x1", df, group_by="group")
    r_all = ols("y ~ x1", data=df)
    html = _html(regtable(r_all, grp))
    assert "(1)" in html
    assert "X" in html


# ── GT-specific tests ─────────────────────────────────────────────


def test_regtable_returns_gt_object(simple_data):
    """regtable() returns a GT instance."""
    r1 = ols("y ~ x1", data=simple_data)
    assert isinstance(regtable(r1), GT)


def test_regtable_repr_html(simple_data):
    """GT object has _repr_html_ for Jupyter rendering."""
    r1 = ols("y ~ x1", data=simple_data)
    html = regtable(r1)._repr_html_()
    assert html is not None
    assert "<table" in html


def test_regtable_as_latex_valid(simple_data):
    """as_latex() produces valid LaTeX with booktabs."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    latex = _latex(regtable(r1))
    assert r"\begin{table}" in latex
    assert r"\toprule" in latex
    assert r"\bottomrule" in latex
    assert "x1" in latex


def test_regtable_latex_footnote(simple_data):
    """LaTeX footnote includes stat description."""
    r1 = ols("y ~ x1", data=simple_data)
    latex = _latex(regtable(r1))
    assert "T-statistics in parentheses" in latex


def test_regtable_latex_auto_escapes(panel_data):
    """LaTeX output escapes underscores in variable names."""
    r1 = ols("y ~ x1 + x2 | firm_id", data=panel_data, cluster=["firm_id"])
    latex = _latex(regtable(r1))
    assert r"firm\_id" in latex


def test_regtable_gt_chainable(simple_data):
    """GT result can be further customized with GT methods."""
    r1 = ols("y ~ x1", data=simple_data)
    table2 = regtable(r1).tab_header(title="Table 1")
    assert isinstance(table2, GT)


def test_regtable_html_fe_indicators(panel_data):
    """HTML output includes FE and cluster indicator rows."""
    r1 = ols("y ~ x1 + x2 | firm_id", data=panel_data, cluster=["firm_id"])
    html = _html(regtable(r1))
    assert "Fixed Effects" in html
    assert "Clustering" in html
    assert "firm_id" in html


def test_regtable_latex_wide(simple_data):
    """LaTeX wide mode uses multicolumn for spanning headers."""
    r1 = ols("y ~ x1", data=simple_data)
    latex = _latex(regtable(r1, wide=True))
    assert r"\multicolumn" in latex or r"\cmidrule" in latex


def test_regtable_mismatched_models(simple_data):
    """Models with different variable sets display correctly."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x2", data=simple_data)
    html = _html(regtable(r1, r2))
    assert "x1" in html
    assert "x2" in html


# ── _normalize_stat unit tests ────────────────────────────────────


def test_normalize_stat_string():
    assert _normalize_stat("t", "round") == [("t", "(", ")")]


def test_normalize_stat_se_square():
    assert _normalize_stat("se", "square") == [("se", "[", "]")]


def test_normalize_stat_tuple():
    assert _normalize_stat(("t", "se"), "round") == [("t", "(", ")"), ("se", "[", "]")]


def test_normalize_stat_tuple_reversed():
    assert _normalize_stat(("se", "t"), "round") == [("se", "(", ")"), ("t", "[", "]")]


def test_normalize_stat_tuple_square():
    assert _normalize_stat(("t", "se"), "square") == [("t", "[", "]"), ("se", "(", ")")]


def test_normalize_stat_none():
    assert _normalize_stat(None, "round") == []


def test_normalize_stat_invalid():
    with pytest.raises(ValueError, match="stat must be"):
        _normalize_stat("p", "round")


def test_normalize_stat_invalid_tuple():
    with pytest.raises(ValueError, match="stat must be"):
        _normalize_stat(("t", "p"), "round")


def test_normalize_stat_too_many():
    with pytest.raises(ValueError, match="1 or 2 elements"):
        _normalize_stat(("t", "se", "p"), "round")


# ── Transposed layout tests ──────────────────────────────────────


def test_regtable_transpose_basic(simple_data):
    """Transposed table has variables as column headers and models as rows."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    html = _html(regtable(r1, r2, transpose=True))
    assert "x1" in html
    assert "(1)" in html
    assert "(2)" in html


def test_regtable_transpose_custom_labels(simple_data):
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    html = _html(regtable(r1, r2, labels=["Base", "Full"], transpose=True))
    assert "Base" in html
    assert "Full" in html


def test_regtable_transpose_stat_se(simple_data):
    r1 = ols("y ~ x1", data=simple_data)
    html = _html(regtable(r1, transpose=True, stat="se"))
    assert "Standard errors in parentheses" in html


def test_regtable_transpose_stat_none(simple_data):
    """No stat sub-rows when stat=None."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, r2, transpose=True, stat=None)
    assert isinstance(table, GT)


def test_regtable_transpose_fe_indicators(simple_data):
    """FE/cluster indicators appear as prefixed columns."""
    fe = np.random.default_rng(42).integers(0, 5, len(simple_data))
    df = simple_data.with_columns(pl.Series("g", fe))
    r1 = ols("y ~ x1 + x2", data=df)
    r2 = ols("y ~ x1 + x2 | g", data=df, cluster=["g"])
    html = _html(regtable(r1, r2, transpose=True))
    assert "FE:g" in html
    assert "Cl:g" in html


def test_regtable_transpose_missing_vars(simple_data):
    """Models with different variables show blanks in transposed layout."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    html = _html(regtable(r1, r2, transpose=True))
    assert "x2" in html


def test_regtable_transpose_latex(simple_data):
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    latex = _latex(regtable(r1, r2, transpose=True))
    assert r"\begin{table}" in latex
    assert r"\toprule" in latex
    assert "x1" in latex
