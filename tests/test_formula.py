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
