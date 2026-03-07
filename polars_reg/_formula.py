from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FormulaSpec:
    depvar: str
    exog: list[str] = field(default_factory=list)
    fe: list[str] = field(default_factory=list)
    endog: list[str] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    add_intercept: bool = True


def parse_formula(formula: str) -> FormulaSpec:
    """Parse a formula string into a FormulaSpec.

    Syntax:
        y ~ x1 + x2                        (OLS)
        y ~ x1 + x2 | fe1 + fe2            (OLS with absorbed FE)
        y ~ x1 | fe1 | x_endog ~ z1 + z2   (IV with FE)
        y ~ x1 + x2 - 1                    (no intercept)
        y ~ x_exog || x_endog ~ z1          (IV, no FE, empty FE slot)
    """
    formula = formula.strip()
    parts = [p.strip() for p in formula.split("|")]

    # First part: depvar ~ exog
    lhs, rhs = parts[0].split("~", 1)
    depvar = lhs.strip()

    add_intercept = True
    rhs = rhs.strip()

    # Check for - 1 or -1 (no intercept)
    if rhs.endswith("- 1") or rhs.endswith("-1"):
        add_intercept = False
        rhs = rhs.rsplit("-", 1)[0].strip().rstrip("+").strip()

    # Parse exog variables
    if rhs in ("1", ""):
        exog: list[str] = []
    else:
        exog = [v.strip() for v in rhs.split("+") if v.strip() and v.strip() != "1"]

    # Second part (optional): fixed effects
    fe: list[str] = []
    if len(parts) >= 2:
        fe_str = parts[1].strip()
        if fe_str:
            fe = [v.strip() for v in fe_str.split("+")]

    # Third part (optional): endogenous ~ instruments
    endog: list[str] = []
    instruments: list[str] = []
    if len(parts) >= 3:
        iv_lhs, iv_rhs = parts[2].split("~", 1)
        endog = [v.strip() for v in iv_lhs.split("+") if v.strip()]
        instruments = [v.strip() for v in iv_rhs.split("+") if v.strip()]

    return FormulaSpec(
        depvar=depvar,
        exog=exog,
        fe=fe,
        endog=endog,
        instruments=instruments,
        add_intercept=add_intercept,
    )
