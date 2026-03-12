"""Tests for regtable (estout-style side-by-side regression table)."""

import numpy as np
import polars as pl
import pytest

from polars_reg import groupby_reg, ols, regtable
from polars_reg._regtable import RegTable, _normalize_stat


def test_regtable_basic(simple_data):
    """Basic regtable with two models produces expected structure."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, r2)
    assert "(1)" in table
    assert "(2)" in table
    assert "x1" in table
    assert "N" in table
    assert "R²" in table


def test_regtable_custom_labels(simple_data):
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, r2, labels=["Base", "Full"])
    assert "Base" in table
    assert "Full" in table


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
    table = regtable(r1, stars=True)
    assert "*" in table


def test_regtable_no_stars(simple_data):
    """No stars when disabled."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, stars=False)
    assert "p<0.10" not in table


# ── stat parameter tests ──────────────────────────────────────────


def test_regtable_default_shows_tstat(simple_data):
    """Default stat='t' shows t-statistics in parentheses."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1)
    assert "T-statistics in parentheses" in table
    # t-stats should be in parens — find a line with parens (not the footnote)
    stat_lines = [
        line
        for line in table.split("\n")
        if "(" in line and ")" in line and "p<" not in line and "T-stat" not in line
    ]
    assert len(stat_lines) > 0


def test_regtable_stat_se(simple_data):
    """stat='se' shows standard errors in parentheses."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, stat="se")
    assert "Standard errors in parentheses" in table


def test_regtable_stat_none(simple_data):
    """stat=None shows coefficients only, no sub-stats."""
    r1 = ols("y ~ x1", data=simple_data)
    table_none = regtable(r1, stat=None)
    table_t = regtable(r1, stat="t")
    # stat=None should have fewer lines than stat="t" (no stat rows)
    assert table_none.count("\n") < table_t.count("\n")
    # No stat footnote
    assert "parentheses" not in table_none
    assert "brackets" not in table_none


def test_regtable_stat_both(simple_data):
    """stat=('t', 'se') shows both t-stats and SEs."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, stat=("t", "se"))
    assert "T-statistics in parentheses" in table
    assert "standard errors in brackets" in table
    # Should have both () and [] stat lines
    lines = table.split("\n")
    paren_lines = [
        ln for ln in lines if ln.strip().startswith("(") and "p<" not in ln and "T-stat" not in ln
    ]
    bracket_lines = [ln for ln in lines if ln.strip().startswith("[")]
    assert len(paren_lines) > 0
    assert len(bracket_lines) > 0


def test_regtable_stat_both_reversed(simple_data):
    """stat=('se', 't') shows SE first in parens, t in brackets."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, stat=("se", "t"))
    assert "Standard errors in parentheses" in table
    assert "t-statistics in brackets" in table


def test_regtable_stat_values_correct(simple_data):
    """Verify actual t-stat and SE values appear in output."""
    r1 = ols("y ~ x1", data=simple_data)
    t_val = r1.tstat[0]  # intercept t-stat
    se_val = r1.se[0]  # intercept SE

    # t-stat table
    t_table = regtable(r1, stat="t")
    t_str = f"{t_val:.4g}"
    assert t_str in t_table

    # SE table
    se_table = regtable(r1, stat="se")
    se_str = f"{se_val:.4g}"
    assert se_str in se_table


# ── brackets parameter tests ──────────────────────────────────────


def test_regtable_brackets_square(simple_data):
    """brackets='square' uses [] as primary brackets."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, brackets="square")
    assert "T-statistics in brackets" in table
    lines = table.split("\n")
    bracket_lines = [ln for ln in lines if ln.strip().startswith("[")]
    assert len(bracket_lines) > 0


def test_regtable_brackets_square_both(simple_data):
    """brackets='square' with both stats: primary in [], secondary in ()."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, stat=("t", "se"), brackets="square")
    assert "T-statistics in brackets" in table
    assert "standard errors in parentheses" in table


def test_regtable_brackets_invalid(simple_data):
    """Invalid brackets value raises ValueError."""
    r1 = ols("y ~ x1", data=simple_data)
    with pytest.raises(ValueError, match="brackets must be"):
        regtable(r1, brackets="curly")


# ── wide parameter tests ──────────────────────────────────────────


def test_regtable_wide(simple_data):
    """wide=True puts stats in columns beside coefficients."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    table_normal = regtable(r1, r2, stat="t")
    table_wide = regtable(r1, r2, stat="t", wide=True)
    # Wide should have fewer lines (no separate stat rows)
    assert table_wide.count("\n") < table_normal.count("\n")


def test_regtable_wide_both(simple_data):
    """wide=True with both stats produces wider rows."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, stat=("t", "se"), wide=True)
    # Should have t-stats and SEs on the same row as coefficients
    lines = table.split("\n")
    x1_line = [ln for ln in lines if ln.startswith("x1")][0]
    # Should contain both () and [] on same line
    assert "(" in x1_line and ")" in x1_line
    assert "[" in x1_line and "]" in x1_line


def test_regtable_wide_stat_none(simple_data):
    """wide=True with stat=None just shows coefficients."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, stat=None, wide=True)
    assert "x1" in table
    # No stat lines
    lines = table.split("\n")
    paren_lines = [
        ln for ln in lines if "(" in ln and ")" in ln and "p<" not in ln and "(1)" not in ln
    ]
    assert len(paren_lines) == 0


# ── Existing feature tests ────────────────────────────────────────


def test_regtable_missing_vars(simple_data):
    """Variables not in a model show as blank."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, r2)
    # x2 row should have blank in column 1
    lines = table.split("\n")
    x2_line = [ln for ln in lines if ln.startswith("x2")][0]
    assert "x2" in x2_line


def test_regtable_fe_indicator_rows(panel_data):
    """Each FE gets its own Y/N indicator row under a 'Fixed Effects' header."""
    r1 = ols("y ~ x1 + x2", data=panel_data)
    r2 = ols("y ~ x1 + x2 | firm_id", data=panel_data)
    r3 = ols("y ~ x1 + x2 | firm_id + year_id", data=panel_data)
    table = regtable(r1, r2, r3)
    assert "Fixed Effects" in table
    lines = table.split("\n")
    firm_line = [ln for ln in lines if "firm_id" in ln and ln.strip().startswith("firm_id")][0]
    year_line = [ln for ln in lines if "year_id" in ln and ln.strip().startswith("year_id")][0]
    assert firm_line.count("Y") == 2
    assert firm_line.count("N") == 1
    assert year_line.count("Y") == 1
    assert year_line.count("N") == 2


def test_regtable_cluster_indicator_rows(panel_data):
    """Each cluster variable gets its own Y/N indicator row under a 'Clustering' header."""
    r1 = ols("y ~ x1 + x2", data=panel_data)
    r2 = ols("y ~ x1 + x2 | firm_id", data=panel_data, cluster=["firm_id"])
    table = regtable(r1, r2)
    assert "Clustering" in table
    lines = table.split("\n")
    cl_line = [ln for ln in lines if "firm_id" in ln and "Clustering" not in ln][-1]
    assert cl_line.count("Y") == 1
    assert cl_line.count("N") == 1


def test_regtable_adj_r2(simple_data):
    """Adj. R² row is present."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1)
    assert "Adj. R²" in table


def test_regtable_precision(simple_data):
    """Different precision values change output."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    t2 = regtable(r1, precision=2)
    t6 = regtable(r1, precision=6)
    assert len(t2) > 0
    assert len(t6) > 0


def test_regtable_groupby_expansion():
    """GroupRegressionResult is auto-expanded with group keys as labels."""
    rng = np.random.default_rng(42)
    n = 300
    group = np.repeat(["A", "B", "C"], n // 3)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "group": group})
    grp = groupby_reg(ols, "y ~ x1", df, group_by="group")
    table = regtable(grp)
    assert "A" in table
    assert "B" in table
    assert "C" in table
    assert "x1" in table


def test_regtable_groupby_mixed(simple_data):
    """Mix of RegressionResult and GroupRegressionResult works."""
    rng = np.random.default_rng(42)
    n = 200
    group = np.repeat(["X", "Y"], n // 2)
    x1 = rng.standard_normal(n)
    y = 1.0 + 2.0 * x1 + rng.standard_normal(n) * 0.5
    df = pl.DataFrame({"y": y, "x1": x1, "group": group})
    grp = groupby_reg(ols, "y ~ x1", df, group_by="group")
    r_all = ols("y ~ x1", data=df)
    table = regtable(r_all, grp)
    assert "(1)" in table
    assert "X" in table


# ── Format-specific tests ─────────────────────────────────────────


def test_regtable_latex(simple_data):
    """LaTeX output has expected structure."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, format="latex")
    assert r"\begin{table}" in table
    assert r"\toprule" in table
    assert r"\midrule" in table
    assert r"\bottomrule" in table
    assert r"\end{tabular}" in table
    assert "x1" in table
    assert "R$^2$" in table


def test_regtable_latex_escapes(panel_data):
    """LaTeX output escapes underscores in variable names."""
    r1 = ols("y ~ x1 + x2 | firm_id", data=panel_data, cluster=["firm_id"])
    table = regtable(r1, format="latex")
    assert r"firm\_id" in table
    assert "firm_id" not in table.split(r"\_")[-1]


def test_regtable_latex_stars(simple_data):
    """LaTeX stars are rendered as superscripts."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, format="latex", stars=True)
    assert "$^{" in table


def test_regtable_latex_footnote(simple_data):
    """LaTeX footnote includes stat description."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, format="latex")
    assert "T-statistics in parentheses" in table


def test_regtable_latex_wide(simple_data):
    """LaTeX wide mode uses multicolumn for spanning headers."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, format="latex", wide=True)
    assert r"\multicolumn" in table


def test_regtable_html(simple_data):
    """HTML output has expected structure."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, format="html")
    assert "<table" in table
    assert "</table>" in table
    assert "<thead>" in table
    assert "<tbody>" in table
    assert "x1" in table
    assert "R&sup2;" in table


def test_regtable_html_fe_indicators(panel_data):
    """HTML output includes FE and cluster indicator rows."""
    r1 = ols("y ~ x1 + x2 | firm_id", data=panel_data, cluster=["firm_id"])
    table = regtable(r1, format="html")
    assert "Fixed Effects" in table
    assert "Clustering" in table
    assert "firm_id" in table


def test_regtable_html_stars(simple_data):
    """HTML stars are rendered as superscripts."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, format="html", stars=True)
    assert "<sup>" in table


def test_regtable_html_footnote(simple_data):
    """HTML footnote includes stat description."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, format="html")
    assert "T-statistics in parentheses" in table


def test_regtable_html_wide(simple_data):
    """HTML wide mode uses colspan."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, format="html", wide=True)
    assert "colspan" in table


# ── RegTable class tests ──────────────────────────────────────────


def test_regtable_returns_regtable_type(simple_data):
    """regtable() returns a RegTable instance (str subclass)."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1)
    assert isinstance(table, RegTable)
    assert isinstance(table, str)


def test_regtable_repr_html_text_mode(simple_data):
    """Text-mode regtable has _repr_html_ returning HTML."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, format="text")
    html = table._repr_html_()
    assert html is not None
    assert "<table" in html
    assert "</table>" in html


def test_regtable_repr_html_html_mode(simple_data):
    """HTML-mode regtable has _repr_html_ returning the same HTML."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, format="html")
    html = table._repr_html_()
    assert html is not None
    assert html == str(table)


def test_regtable_repr_html_latex_mode(simple_data):
    """LaTeX-mode regtable has no _repr_html_ (returns None)."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1, format="latex")
    assert table._repr_html_() is None


def test_regtable_str_operations(simple_data):
    """RegTable works like a regular string for concatenation, slicing, etc."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1)
    combined = table + "\n\nAdditional notes"
    assert "Additional notes" in combined
    assert table.startswith("=")
    assert len(table) > 0


# ── Additional robustness tests ───────────────────────────────────


def test_regtable_mismatched_models(simple_data):
    """Models with different variable sets display correctly."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x2", data=simple_data)
    table = regtable(r1, r2)
    assert "x1" in table
    assert "x2" in table
    lines = table.split("\n")
    x1_line = [ln for ln in lines if ln.startswith("x1")][0]
    x2_line = [ln for ln in lines if ln.startswith("x2")][0]
    assert "x1" in x1_line
    assert "x2" in x2_line


# ── _normalize_stat unit tests ────────────────────────────────────


def test_normalize_stat_string():
    specs = _normalize_stat("t", "round")
    assert specs == [("t", "(", ")")]


def test_normalize_stat_se_square():
    specs = _normalize_stat("se", "square")
    assert specs == [("se", "[", "]")]


def test_normalize_stat_tuple():
    specs = _normalize_stat(("t", "se"), "round")
    assert specs == [("t", "(", ")"), ("se", "[", "]")]


def test_normalize_stat_tuple_reversed():
    specs = _normalize_stat(("se", "t"), "round")
    assert specs == [("se", "(", ")"), ("t", "[", "]")]


def test_normalize_stat_tuple_square():
    specs = _normalize_stat(("t", "se"), "square")
    assert specs == [("t", "[", "]"), ("se", "(", ")")]


def test_normalize_stat_none():
    specs = _normalize_stat(None, "round")
    assert specs == []


def test_normalize_stat_invalid():
    with pytest.raises(ValueError, match="stat must be"):
        _normalize_stat("p", "round")


def test_normalize_stat_invalid_tuple():
    with pytest.raises(ValueError, match="stat must be"):
        _normalize_stat(("t", "p"), "round")


def test_normalize_stat_too_many():
    with pytest.raises(ValueError, match="1 or 2 elements"):
        _normalize_stat(("t", "se", "p"), "round")
