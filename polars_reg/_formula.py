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
    indicators: set[str] = field(default_factory=set)
    """Column names that should be expanded as indicator (dummy) variables."""


def _expand_star(term: str) -> list[str]:
    """Expand x1*x2 into [x1, x2, x1:x2]. Handles arbitrary order."""
    if "*" not in term:
        return [term]
    parts = [p.strip() for p in term.split("*")]
    result: list[str] = []
    # All non-empty subsets of parts, ordered by size
    for size in range(1, len(parts) + 1):
        from itertools import combinations

        for subset in combinations(parts, size):
            result.append(":".join(subset))
    return result


def _strip_indicator(term: str) -> tuple[str, bool]:
    """Strip i. prefix from a term. Returns (clean_name, is_indicator)."""
    if term.startswith("i."):
        return term[2:], True
    return term, False


def parse_formula(formula: str) -> FormulaSpec:
    """Parse a formula string into a FormulaSpec.

    The formula uses ``|`` to separate up to three parts:

    1. **Dependent variable ~ exogenous regressors**
    2. **Fixed effects** (optional, absorbed)
    3. **Endogenous ~ instruments** (optional, IV)

    Supported syntaxes
    ------------------
    ``y ~ x1 + x2``
        OLS with intercept.
    ``y ~ x1 + x2 - 1``
        OLS without intercept.
    ``y ~ x1 + x2 | fe1 + fe2``
        OLS with absorbed fixed effects (reghdfe-style).
    ``y ~ x1 | fe1 | endog ~ z1 + z2``
        IV (2SLS / LIML / GMM) with one FE and an endogenous variable
        instrumented by *z1* and *z2* (fixest-style three-part formula).
    ``y ~ x1 || endog ~ z1 + z2``
        IV without fixed effects.  The double-pipe ``||`` is shorthand for
        an empty FE slot: it is normalised to ``| |`` internally so the
        parser sees three pipe-separated parts with an empty second part.
    ``y ~ x1 + x2 | | endog ~ z1``
        Equivalent to the ``||`` form above — the FE slot is explicitly
        empty (single space between pipes).
    ``y ~ x1*x2``
        Interaction expansion: produces ``x1 + x2 + x1:x2``.
    ``y ~ x1:x2``
        Interaction term only (no main effects).
    ``y ~ i.group + x1``
        Indicator (dummy) variable expansion for *group*.
    ``y ~ i.group*x``
        Indicator variable with full interaction expansion.
    """
    formula = formula.strip()

    # Normalise double-pipe shorthand: "||" → "| |" so the split on "|"
    # produces an explicit empty FE slot between the two pipes.
    formula = formula.replace("||", "| |")

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

    # Parse exog variables, expanding * interactions and i. indicators
    indicators: set[str] = set()
    if rhs in ("1", ""):
        exog: list[str] = []
    else:
        raw_terms = [v.strip() for v in rhs.split("+") if v.strip() and v.strip() != "1"]
        exog = []
        for term in raw_terms:
            # Strip i. prefix before expanding * (e.g. i.x1*x2)
            # Process each sub-part of * for i. prefix
            if "*" in term:
                sub_parts = [p.strip() for p in term.split("*")]
                clean_parts = []
                for sp in sub_parts:
                    clean, is_ind = _strip_indicator(sp)
                    if is_ind:
                        indicators.add(clean)
                    clean_parts.append(clean)
                term = "*".join(clean_parts)
            elif ":" in term:
                # Handle i. prefix inside colon terms (e.g. i.group:x)
                colon_parts = term.split(":")
                clean_colon = []
                for cp in colon_parts:
                    clean, is_ind = _strip_indicator(cp.strip())
                    if is_ind:
                        indicators.add(clean)
                    clean_colon.append(clean)
                term = ":".join(clean_colon)
            else:
                term, is_ind = _strip_indicator(term)
                if is_ind:
                    indicators.add(term)
            for expanded in _expand_star(term):
                if expanded not in exog:
                    exog.append(expanded)

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
        indicators=indicators,
    )
