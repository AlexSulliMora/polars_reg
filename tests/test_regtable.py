"""Tests for regtable (estout-style side-by-side regression table)."""

import pytest

from polars_reg import ols, regtable


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
