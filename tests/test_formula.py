import pytest

from polars_reg._formula import parse_formula


def test_simple_formula():
    spec = parse_formula("y ~ x1 + x2")
    assert spec.depvar == "y"
    assert spec.exog == ["x1", "x2"]
    assert spec.fe == []
    assert spec.endog == []
    assert spec.instruments == []
    assert spec.add_intercept is True


def test_no_intercept():
    spec = parse_formula("y ~ x1 + x2 - 1")
    assert spec.add_intercept is False
    assert spec.exog == ["x1", "x2"]


def test_intercept_only():
    spec = parse_formula("y ~ 1")
    assert spec.exog == []
    assert spec.add_intercept is True


def test_fe_formula():
    spec = parse_formula("y ~ x1 + x2 | firm_id + year_id")
    assert spec.depvar == "y"
    assert spec.exog == ["x1", "x2"]
    assert spec.fe == ["firm_id", "year_id"]
    assert spec.add_intercept is True


def test_iv_formula():
    spec = parse_formula("y ~ x_exog | firm_id | x_endog ~ z1 + z2")
    assert spec.depvar == "y"
    assert spec.exog == ["x_exog"]
    assert spec.fe == ["firm_id"]
    assert spec.endog == ["x_endog"]
    assert spec.instruments == ["z1", "z2"]


def test_iv_no_fe():
    spec = parse_formula("y ~ x_exog || x_endog ~ z1")
    assert spec.fe == []
    assert spec.endog == ["x_endog"]
    assert spec.instruments == ["z1"]


def test_interaction_colon():
    spec = parse_formula("y ~ x1 + x2 + x1:x2")
    assert spec.exog == ["x1", "x2", "x1:x2"]


def test_interaction_star():
    """x1*x2 expands to x1 + x2 + x1:x2."""
    spec = parse_formula("y ~ x1*x2")
    assert spec.exog == ["x1", "x2", "x1:x2"]


def test_interaction_star_with_other_vars():
    spec = parse_formula("y ~ x0 + x1*x2")
    assert spec.exog == ["x0", "x1", "x2", "x1:x2"]


def test_three_way_interaction():
    """x1*x2*x3 expands to all subsets."""
    spec = parse_formula("y ~ x1*x2*x3")
    assert "x1" in spec.exog
    assert "x2" in spec.exog
    assert "x3" in spec.exog
    assert "x1:x2" in spec.exog
    assert "x1:x3" in spec.exog
    assert "x2:x3" in spec.exog
    assert "x1:x2:x3" in spec.exog
    assert len(spec.exog) == 7


def test_interaction_no_duplicates():
    """x1*x2 + x1 should not duplicate x1."""
    spec = parse_formula("y ~ x1*x2 + x1")
    assert spec.exog.count("x1") == 1


def test_interaction_with_fe():
    spec = parse_formula("y ~ x1*x2 | fe1")
    assert spec.exog == ["x1", "x2", "x1:x2"]
    assert spec.fe == ["fe1"]


# ── Indicator variables ──────────────────────────────────────────


def test_indicator_basic():
    spec = parse_formula("y ~ x + i.group")
    assert spec.exog == ["x", "group"]
    assert "group" in spec.indicators


def test_indicator_star():
    """i.group*x expands with group marked as indicator."""
    spec = parse_formula("y ~ i.group*x")
    assert "group" in spec.indicators
    assert "x" not in spec.indicators
    assert spec.exog == ["group", "x", "group:x"]


def test_indicator_colon():
    """i.group:x marks group as indicator in colon term."""
    spec = parse_formula("y ~ x + i.group:x")
    assert "group" in spec.indicators
    assert spec.exog == ["x", "group:x"]


def test_indicator_multiple():
    spec = parse_formula("y ~ i.a + i.b + x")
    assert spec.indicators == {"a", "b"}
    assert spec.exog == ["a", "b", "x"]


# ── Fixest-style IV syntax ──────────────────────────────────────


def test_iv_with_fe_fixest_style():
    """y ~ x1 | fe1 | endog ~ z1 + z2 (three-part fixest-style)."""
    spec = parse_formula("y ~ x1 | fe1 | endog ~ z1 + z2")
    assert spec.depvar == "y"
    assert spec.exog == ["x1"]
    assert spec.fe == ["fe1"]
    assert spec.endog == ["endog"]
    assert spec.instruments == ["z1", "z2"]
    assert spec.add_intercept is True


def test_iv_double_pipe_no_fe():
    """y ~ x1 || endog ~ z1 + z2 (double-pipe shorthand, no FE)."""
    spec = parse_formula("y ~ x1 || endog ~ z1 + z2")
    assert spec.depvar == "y"
    assert spec.exog == ["x1"]
    assert spec.fe == []
    assert spec.endog == ["endog"]
    assert spec.instruments == ["z1", "z2"]


def test_iv_double_pipe_multiple_exog():
    """y ~ x1 + x2 || endog ~ z1 + z2 (double-pipe, multiple exog)."""
    spec = parse_formula("y ~ x1 + x2 || endog ~ z1 + z2")
    assert spec.depvar == "y"
    assert spec.exog == ["x1", "x2"]
    assert spec.fe == []
    assert spec.endog == ["endog"]
    assert spec.instruments == ["z1", "z2"]


def test_iv_explicit_empty_fe_slot():
    """y ~ x1 | | endog ~ z1 (explicit empty FE slot with space)."""
    spec = parse_formula("y ~ x1 | | endog ~ z1")
    assert spec.fe == []
    assert spec.endog == ["endog"]
    assert spec.instruments == ["z1"]


def test_double_pipe_and_pipe_space_pipe_equivalent():
    """|| and | | must produce identical FormulaSpec."""
    spec_double = parse_formula("y ~ x1 + x2 || endog ~ z1 + z2")
    spec_spaced = parse_formula("y ~ x1 + x2 | | endog ~ z1 + z2")

    assert spec_double.depvar == spec_spaced.depvar
    assert spec_double.exog == spec_spaced.exog
    assert spec_double.fe == spec_spaced.fe
    assert spec_double.endog == spec_spaced.endog
    assert spec_double.instruments == spec_spaced.instruments
    assert spec_double.add_intercept == spec_spaced.add_intercept


def test_iv_multiple_endog():
    """Multiple endogenous variables with multiple instruments."""
    spec = parse_formula("y ~ x1 | fe1 | endog1 + endog2 ~ z1 + z2 + z3")
    assert spec.endog == ["endog1", "endog2"]
    assert spec.instruments == ["z1", "z2", "z3"]
    assert spec.fe == ["fe1"]


def test_iv_multiple_fe():
    """IV with two fixed effects."""
    spec = parse_formula("y ~ x1 | fe1 + fe2 | endog ~ z1")
    assert spec.fe == ["fe1", "fe2"]
    assert spec.endog == ["endog"]
    assert spec.instruments == ["z1"]


# ── OLS with single / multiple FE ──────────────────────────────


def test_ols_single_fe():
    """y ~ x1 + x2 | fe1 (OLS with one FE)."""
    spec = parse_formula("y ~ x1 + x2 | fe1")
    assert spec.exog == ["x1", "x2"]
    assert spec.fe == ["fe1"]
    assert spec.endog == []
    assert spec.instruments == []


def test_ols_two_fe():
    """y ~ x1 | fe1 + fe2 (OLS with two FE)."""
    spec = parse_formula("y ~ x1 | fe1 + fe2")
    assert spec.exog == ["x1"]
    assert spec.fe == ["fe1", "fe2"]


# ── Edge-case / whitespace handling ─────────────────────────────


def test_whitespace_tolerance():
    """Extra whitespace should be ignored."""
    spec = parse_formula("  y  ~  x1 + x2  |  fe1  |  endog ~ z1  ")
    assert spec.depvar == "y"
    assert spec.exog == ["x1", "x2"]
    assert spec.fe == ["fe1"]
    assert spec.endog == ["endog"]
    assert spec.instruments == ["z1"]


def test_no_intercept_with_fe():
    """No-intercept flag with fixed effects."""
    spec = parse_formula("y ~ x1 + x2 - 1 | fe1")
    assert spec.add_intercept is False
    assert spec.exog == ["x1", "x2"]
    assert spec.fe == ["fe1"]


def test_no_intercept_variant_spacing():
    """'-1' without space before 1."""
    spec = parse_formula("y ~ x1 + x2 -1")
    assert spec.add_intercept is False
    assert spec.exog == ["x1", "x2"]


# ── Error-path tests ─────────────────────────────────────────────


def test_parse_formula_missing_tilde():
    with pytest.raises((ValueError, IndexError)):
        parse_formula("y x1 + x2")


def test_parse_formula_empty_lhs():
    """Empty LHS produces an empty depvar string."""
    spec = parse_formula("~ x1 + x2")
    assert spec.depvar == ""
