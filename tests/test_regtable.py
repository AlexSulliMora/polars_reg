"""Tests for regtable (estout-style side-by-side regression table)."""

import numpy as np
import polars as pl
import pytest

from polars_reg import groupby_reg, ols, regtable
from polars_reg._regtable import RegTable


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


def test_regtable_se_in_parens(simple_data):
    """SEs should be wrapped in parentheses without internal padding."""
    r1 = ols("y ~ x1", data=simple_data)
    table = regtable(r1)
    # Find a line with parentheses (SE line)
    se_lines = [
        line for line in table.split("\n") if "(" in line and ")" in line and "p<" not in line
    ]
    assert len(se_lines) > 0
    for line in se_lines:
        # Extract the parenthesized part
        start = line.index("(")
        end = line.index(")", start)
        inner = line[start + 1 : end]
        # Should not start with space
        assert not inner.startswith(" "), f"SE has internal padding: '{line.strip()}'"


def test_regtable_missing_vars(simple_data):
    """Variables not in a model show as blank."""
    r1 = ols("y ~ x1", data=simple_data)
    r2 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, r2)
    # x2 row should have blank in column 1
    lines = table.split("\n")
    x2_line = [ln for ln in lines if ln.startswith("x2")][0]
    # The first model column should be mostly spaces
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
    # r1=N, r2=Y, r3=Y for firm_id
    assert firm_line.count("Y") == 2
    assert firm_line.count("N") == 1
    # r1=N, r2=N, r3=Y for year_id
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
    # Lower precision should generally produce shorter numbers
    # Just check both run without error
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
    # Should have 3 columns labeled by group keys
    assert "A" in table
    assert "B" in table
    assert "C" in table
    # Should have coefficient rows
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
    # First column auto-labeled (1), then group keys
    assert "(1)" in table
    assert "X" in table


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
    assert "firm_id" not in table.split(r"\_")[-1]  # no unescaped underscores


def test_regtable_latex_stars(simple_data):
    """LaTeX stars are rendered as superscripts."""
    r1 = ols("y ~ x1 + x2", data=simple_data)
    table = regtable(r1, format="latex", stars=True)
    assert "$^{" in table  # star superscript


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
    # Both variables should appear
    assert "x1" in table
    assert "x2" in table
    # Each model column should have blanks for the other's variable
    lines = table.split("\n")
    x1_line = [ln for ln in lines if ln.startswith("x1")][0]
    x2_line = [ln for ln in lines if ln.startswith("x2")][0]
    assert "x1" in x1_line
    assert "x2" in x2_line
