from polars_reg._formula import FormulaSpec, parse_formula


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
