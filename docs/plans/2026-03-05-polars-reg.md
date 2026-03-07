# polars_reg Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Polars-native Python package implementing OLS, reghdfe-style FE absorption, IV/2SLS, LIML, GMM, and panel estimators with multi-way clustered standard errors.

**Architecture:** Formula string → parse into spec → extract Polars columns to NumPy → demean if FE → estimate coefficients → compute variance-covariance matrix → package into result object. All data handling via Polars; all linear algebra via NumPy/SciPy.

**Tech Stack:** Python 3.11+, polars, numpy, scipy, pytest, ruff, mypy

---

## Phase 1: Project Scaffolding

### Task 1.1: pyproject.toml and package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `polars_reg/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "polars_reg"
version = "0.1.0"
description = "Econometric regression methods using Polars DataFrames"
requires-python = ">=3.11"
dependencies = [
    "polars>=1.0",
    "numpy>=1.24",
    "scipy>=1.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.4",
    "mypy>=1.0",
]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.mypy]
python_version = "3.11"
strict = true
```

**Step 2: Create package __init__.py**

```python
"""polars_reg: Econometric regression methods using Polars DataFrames."""
```

**Step 3: Create tests/__init__.py (empty) and tests/conftest.py**

```python
import numpy as np
import polars as pl
import pytest


@pytest.fixture
def simple_data() -> pl.DataFrame:
    """Simple dataset for basic OLS tests. Known Stata results:
    reg y x1 x2 => b_x1=1.5, b_x2=-0.5, b_cons=2.0 (approx, from DGP)
    """
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    e = rng.standard_normal(n) * 0.5
    y = 2.0 + 1.5 * x1 - 0.5 * x2 + e
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture
def panel_data() -> pl.DataFrame:
    """Panel dataset with firm/year FE for reghdfe-style tests."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 50, 20
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(n_years), n_firms)
    firm_fe = rng.standard_normal(n_firms)
    year_fe = rng.standard_normal(n_years)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    e = rng.standard_normal(n) * 0.5
    y = 1.0 * x1 - 2.0 * x2 + firm_fe[firm_id] + year_fe[year_id] + e
    return pl.DataFrame({
        "y": y, "x1": x1, "x2": x2,
        "firm_id": firm_id, "year_id": year_id,
    })


@pytest.fixture
def iv_data() -> pl.DataFrame:
    """IV dataset with endogenous regressor and instruments."""
    rng = np.random.default_rng(42)
    n = 1000
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    x_exog = rng.standard_normal(n)
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + u
    return pl.DataFrame({
        "y": y, "x_endog": x_endog, "x_exog": x_exog,
        "z1": z1, "z2": z2,
    })
```

**Step 4: Install and verify**

Run: `uv pip install -e ".[dev]" && pytest --co -q`
Expected: "no tests ran" (collected 0 items)

**Step 5: Commit**

```bash
git init && git add pyproject.toml polars_reg/ tests/
git commit -m "feat: project scaffolding with pyproject.toml and test fixtures"
```

---

## Phase 2: Formula Parser

### Task 2.1: Basic formula parsing (y ~ x1 + x2)

**Files:**
- Create: `polars_reg/_formula.py`
- Create: `tests/test_formula.py`

**Step 1: Write the failing test**

```python
# tests/test_formula.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_formula.py -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# polars_reg/_formula.py
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
    """
    formula = formula.strip()

    # Split on |
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_formula.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add polars_reg/_formula.py tests/test_formula.py
git commit -m "feat: formula parser for basic OLS syntax"
```

### Task 2.2: Formula parsing with FE and IV syntax

**Files:**
- Modify: `tests/test_formula.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run tests**

Run: `pytest tests/test_formula.py -v`
Expected: Should already PASS if implementation handles these cases. If `test_iv_no_fe` fails (empty string between `||`), fix the edge case.

**Step 3: Commit**

```bash
git add tests/test_formula.py
git commit -m "test: formula parsing for FE and IV syntax"
```

---

## Phase 3: Utilities

### Task 3.1: Polars-to-NumPy extraction

**Files:**
- Create: `polars_reg/_utils.py`
- Create: `tests/test_utils.py`

**Step 1: Write the failing test**

```python
# tests/test_utils.py
import numpy as np
import polars as pl

from polars_reg._formula import FormulaSpec
from polars_reg._utils import extract_arrays


def test_extract_basic():
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x1": [4.0, 5.0, 6.0]})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.y.shape == (3,)
    assert arrays.X.shape == (3, 2)  # x1 + intercept
    np.testing.assert_array_equal(arrays.y, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(arrays.X[:, 0], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(arrays.X[:, 1], [1.0, 1.0, 1.0])  # intercept last
    assert arrays.names == ["x1", "_cons"]


def test_extract_no_intercept():
    df = pl.DataFrame({"y": [1.0, 2.0], "x1": [3.0, 4.0]})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=False)
    arrays = extract_arrays(df, spec)
    assert arrays.X.shape == (2, 1)
    assert arrays.names == ["x1"]


def test_extract_drops_na():
    df = pl.DataFrame({"y": [1.0, None, 3.0], "x1": [4.0, 5.0, 6.0]})
    spec = FormulaSpec(depvar="y", exog=["x1"], add_intercept=True)
    arrays = extract_arrays(df, spec)
    assert arrays.y.shape == (2,)
    assert arrays.n_obs == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils.py -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# polars_reg/_utils.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from polars_reg._formula import FormulaSpec


@dataclass
class ExtractedArrays:
    y: np.ndarray
    X: np.ndarray
    names: list[str]
    n_obs: int
    fe_arrays: dict[str, np.ndarray]  # fe_name -> integer codes
    cluster_arrays: dict[str, np.ndarray]  # cluster_name -> integer codes
    endog: np.ndarray | None = None
    instruments: np.ndarray | None = None
    endog_names: list[str] | None = None
    instrument_names: list[str] | None = None


def extract_arrays(
    df: pl.DataFrame | pl.LazyFrame,
    spec: FormulaSpec,
    cluster: list[str] | None = None,
) -> ExtractedArrays:
    """Extract NumPy arrays from a Polars DataFrame given a FormulaSpec."""
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Determine all columns needed
    all_cols = [spec.depvar] + spec.exog + spec.fe + spec.endog + spec.instruments
    if cluster:
        all_cols += [c for c in cluster if c not in all_cols]
    all_cols = list(dict.fromkeys(all_cols))  # dedupe preserving order

    # Drop rows with nulls in numeric columns
    numeric_cols = [spec.depvar] + spec.exog + spec.endog + spec.instruments
    df_clean = df.select(all_cols).drop_nulls(subset=numeric_cols)

    n_obs = len(df_clean)

    # Extract y
    y = df_clean[spec.depvar].to_numpy(allow_copy=False).astype(np.float64)

    # Extract X with optional intercept
    names: list[str] = []
    x_cols: list[np.ndarray] = []
    for col in spec.exog:
        x_cols.append(df_clean[col].to_numpy(allow_copy=False).astype(np.float64))
        names.append(col)
    if spec.add_intercept:
        x_cols.append(np.ones(n_obs, dtype=np.float64))
        names.append("_cons")
    X = np.column_stack(x_cols) if x_cols else np.empty((n_obs, 0), dtype=np.float64)

    # Extract FE as integer codes
    fe_arrays: dict[str, np.ndarray] = {}
    for col in spec.fe:
        codes = df_clean[col].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
        fe_arrays[col] = codes.astype(np.int32)

    # Extract cluster codes
    cluster_arrays: dict[str, np.ndarray] = {}
    if cluster:
        for col in cluster:
            codes = df_clean[col].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
            cluster_arrays[col] = codes.astype(np.int32)

    # Extract endogenous and instruments
    endog = None
    instruments = None
    endog_names = None
    instrument_names = None
    if spec.endog:
        endog_cols = [df_clean[c].to_numpy(allow_copy=False).astype(np.float64) for c in spec.endog]
        endog = np.column_stack(endog_cols)
        endog_names = list(spec.endog)
    if spec.instruments:
        iv_cols = [df_clean[c].to_numpy(allow_copy=False).astype(np.float64) for c in spec.instruments]
        instruments = np.column_stack(iv_cols)
        instrument_names = list(spec.instruments)

    return ExtractedArrays(
        y=y, X=X, names=names, n_obs=n_obs,
        fe_arrays=fe_arrays, cluster_arrays=cluster_arrays,
        endog=endog, instruments=instruments,
        endog_names=endog_names, instrument_names=instrument_names,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add polars_reg/_utils.py tests/test_utils.py
git commit -m "feat: Polars-to-NumPy array extraction utility"
```

---

## Phase 4: Standard Errors

### Task 4.1: Homoskedastic and robust (HC0-HC3) standard errors

**Files:**
- Create: `polars_reg/_se.py`
- Create: `tests/test_se.py`

**Step 1: Write the failing test**

```python
# tests/test_se.py
import numpy as np
import pytest

from polars_reg._se import vcov_iid, vcov_robust


def _make_ols_data():
    """Simple OLS: y = 2 + 3*x + e, return X, y, residuals, beta."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.standard_normal(n)
    e = rng.standard_normal(n)
    X = np.column_stack([x, np.ones(n)])
    y = 2.0 + 3.0 * x + e
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    return X, y, resid, beta, XtX_inv


def test_vcov_iid():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    n, k = X.shape
    V = vcov_iid(X, resid)
    sigma2 = resid @ resid / (n - k)
    expected = sigma2 * XtX_inv
    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_vcov_hc1():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    n, k = X.shape
    V = vcov_robust(X, resid, kind="HC1")
    # HC1: (n/(n-k)) * (X'X)^{-1} X' diag(e^2) X (X'X)^{-1}
    meat = X.T @ np.diag(resid**2) @ X
    expected = (n / (n - k)) * XtX_inv @ meat @ XtX_inv
    np.testing.assert_allclose(V, expected, rtol=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_se.py -v`
Expected: FAIL (ImportError)

**Step 3: Write minimal implementation**

```python
# polars_reg/_se.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def vcov_iid(X: NDArray, resid: NDArray) -> NDArray:
    """Homoskedastic variance-covariance: sigma^2 * (X'X)^{-1}."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    sigma2 = resid @ resid / (n - k)
    return sigma2 * XtX_inv


def vcov_robust(X: NDArray, resid: NDArray, kind: str = "HC1") -> NDArray:
    """Heteroskedasticity-robust VCV (HC0, HC1, HC2, HC3)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    if kind == "HC0":
        weights = resid**2
    elif kind == "HC1":
        weights = resid**2 * (n / (n - k))
    elif kind == "HC2":
        hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)  # diagonal of hat matrix
        weights = resid**2 / (1.0 - hat)
    elif kind == "HC3":
        hat = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        weights = resid**2 / (1.0 - hat) ** 2
    else:
        raise ValueError(f"Unknown robust SE kind: {kind}")

    if kind == "HC1":
        meat = X.T @ np.diag(resid**2) @ X
        return (n / (n - k)) * XtX_inv @ meat @ XtX_inv

    meat = X.T @ (X * weights[:, None])
    return XtX_inv @ meat @ XtX_inv
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_se.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polars_reg/_se.py tests/test_se.py
git commit -m "feat: homoskedastic and HC0-HC3 robust standard errors"
```

### Task 4.2: One-way clustered standard errors

**Files:**
- Modify: `polars_reg/_se.py`
- Modify: `tests/test_se.py`

**Step 1: Write the failing test**

```python
def test_vcov_clustered_oneway():
    X, y, resid, beta, XtX_inv = _make_ols_data()
    n = len(resid)
    # Create 10 clusters of 10 obs each
    clusters = np.repeat(np.arange(10), 10)
    V = vcov_clustered(X, resid, clusters)
    # Manual: sum score vectors within clusters, form meat
    G = 10
    k = X.shape[1]
    score = X * resid[:, None]
    meat = np.zeros((k, k))
    for g in range(G):
        mask = clusters == g
        sg = score[mask].sum(axis=0)
        meat += np.outer(sg, sg)
    dfc = (G / (G - 1)) * ((n - 1) / (n - k))
    expected = dfc * XtX_inv @ meat @ XtX_inv
    np.testing.assert_allclose(V, expected, rtol=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_se.py::test_vcov_clustered_oneway -v`
Expected: FAIL (ImportError for `vcov_clustered`)

**Step 3: Implement one-way clustering**

Add to `polars_reg/_se.py`:

```python
def _clustered_meat(X: NDArray, resid: NDArray, clusters: NDArray) -> NDArray:
    """Compute the meat of a clustered sandwich: sum_g (s_g s_g')."""
    k = X.shape[1]
    score = X * resid[:, None]
    unique_clusters = np.unique(clusters)
    meat = np.zeros((k, k))
    for g in unique_clusters:
        mask = clusters == g
        sg = score[mask].sum(axis=0)
        meat += np.outer(sg, sg)
    return meat


def vcov_clustered(
    X: NDArray,
    resid: NDArray,
    clusters: NDArray,
    df_correction: bool = True,
) -> NDArray:
    """One-way cluster-robust VCV (CRV1)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = _clustered_meat(X, resid, clusters)
    G = len(np.unique(clusters))
    if df_correction:
        dfc = (G / (G - 1)) * ((n - 1) / (n - k))
    else:
        dfc = 1.0
    return dfc * XtX_inv @ meat @ XtX_inv
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_se.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polars_reg/_se.py tests/test_se.py
git commit -m "feat: one-way clustered standard errors (CRV1)"
```

### Task 4.3: Multi-way clustered standard errors (Cameron-Gelbach-Miller)

**Files:**
- Modify: `polars_reg/_se.py`
- Modify: `tests/test_se.py`

**Step 1: Write the failing test**

```python
from polars_reg._se import vcov_multiway_clustered


def test_vcov_twoway_clustered():
    """Two-way clustering: V = V_A + V_B - V_{A*B}."""
    rng = np.random.default_rng(42)
    n = 200
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    # 10 firms, 20 years
    firm = np.repeat(np.arange(10), 20)
    year = np.tile(np.arange(20), 10)

    V = vcov_multiway_clustered(X, resid, [firm, year])

    # Manual: V_firm + V_year - V_firm_x_year
    V_firm = vcov_clustered(X, resid, firm, df_correction=True)
    V_year = vcov_clustered(X, resid, year, df_correction=True)
    # Intersection clusters
    interaction = firm * 100 + year
    V_inter = vcov_clustered(X, resid, interaction, df_correction=True)
    expected = V_firm + V_year - V_inter

    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_vcov_threeway_clustered():
    """Three-way: V = V_A + V_B + V_C - V_AB - V_AC - V_BC + V_ABC."""
    rng = np.random.default_rng(42)
    n = 120
    X = np.column_stack([rng.standard_normal(n), np.ones(n)])
    resid = rng.standard_normal(n)
    a = np.repeat(np.arange(4), 30)
    b = np.tile(np.repeat(np.arange(6), 5), 4)
    c = np.tile(np.arange(5), 24)

    V = vcov_multiway_clustered(X, resid, [a, b, c])
    assert V.shape == (2, 2)
    # Just check it runs and is symmetric
    np.testing.assert_allclose(V, V.T, atol=1e-14)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_se.py::test_vcov_twoway_clustered -v`
Expected: FAIL (ImportError)

**Step 3: Implement CGM multi-way clustering**

Add to `polars_reg/_se.py`:

```python
from itertools import combinations


def _interaction_codes(*arrays: NDArray) -> NDArray:
    """Create unique integer codes for the interaction of multiple cluster arrays."""
    if len(arrays) == 1:
        return arrays[0]
    # Use structured array for unique combinations
    n = len(arrays[0])
    dtype = [(f"f{i}", arr.dtype) for i, arr in enumerate(arrays)]
    structured = np.empty(n, dtype=dtype)
    for i, arr in enumerate(arrays):
        structured[f"f{i}"] = arr
    _, codes = np.unique(structured, return_inverse=True)
    return codes


def vcov_multiway_clustered(
    X: NDArray,
    resid: NDArray,
    cluster_list: list[NDArray],
) -> NDArray:
    """Multi-way clustered VCV via Cameron-Gelbach-Miller inclusion-exclusion.

    V = sum over non-empty subsets S of (-1)^(|S|+1) * V_S
    where V_S is one-way clustered VCV using intersection of dimensions in S.
    """
    D = len(cluster_list)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    V = np.zeros((k, k))
    dims = list(range(D))

    for size in range(1, D + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(dims, size):
            # Form intersection cluster
            subset_arrays = [cluster_list[d] for d in subset]
            interaction = _interaction_codes(*subset_arrays)
            G = len(np.unique(interaction))
            meat = _clustered_meat(X, resid, interaction)
            dfc = (G / (G - 1)) * ((n - 1) / (n - k))
            V += sign * dfc * XtX_inv @ meat @ XtX_inv

    return V
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_se.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polars_reg/_se.py tests/test_se.py
git commit -m "feat: multi-way clustered SEs via Cameron-Gelbach-Miller inclusion-exclusion"
```

---

## Phase 5: OLS Estimator + Results Object

### Task 5.1: Results dataclass

**Files:**
- Create: `polars_reg/_results.py`

**Step 1: Write the failing test**

```python
# tests/test_results.py
import numpy as np

from polars_reg._results import RegressionResult


def test_result_basic():
    beta = np.array([1.5, 2.0])
    vcov = np.diag([0.01, 0.04])
    r = RegressionResult(
        coefficients=beta, vcov=vcov, residuals=np.zeros(100),
        names=["x1", "_cons"], n_obs=100, k=2,
        df_r=98, r_squared=0.85, r_squared_adj=0.84,
        model_type="OLS", vcov_type="iid",
    )
    np.testing.assert_allclose(r.se, [0.1, 0.2])
    np.testing.assert_allclose(r.tstat, [15.0, 10.0])
    assert r.summary() is not None  # returns a string
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_results.py -v`

**Step 3: Implement**

```python
# polars_reg/_results.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class RegressionResult:
    coefficients: NDArray
    vcov: NDArray
    residuals: NDArray
    names: list[str]
    n_obs: int
    k: int
    df_r: int
    r_squared: float
    r_squared_adj: float
    model_type: str
    vcov_type: str
    f_stat: float | None = None
    f_pvalue: float | None = None
    n_clusters: dict[str, int] | None = None
    fe_absorbed: list[str] | None = None
    df_absorbed: int = 0

    @property
    def se(self) -> NDArray:
        return np.sqrt(np.diag(self.vcov))

    @property
    def tstat(self) -> NDArray:
        return self.coefficients / self.se

    @property
    def pvalue(self) -> NDArray:
        return 2.0 * stats.t.sf(np.abs(self.tstat), df=self.df_r)

    def confint(self, alpha: float = 0.05) -> NDArray:
        """Return (k, 2) array of [lower, upper] confidence intervals."""
        t_crit = stats.t.ppf(1 - alpha / 2, df=self.df_r)
        margin = t_crit * self.se
        return np.column_stack([self.coefficients - margin, self.coefficients + margin])

    def summary(self) -> str:
        lines = [
            f"{self.model_type} Regression",
            f"{'='*60}",
            f"N = {self.n_obs}    R² = {self.r_squared:.4f}    Adj. R² = {self.r_squared_adj:.4f}",
            f"SE type: {self.vcov_type}",
        ]
        if self.fe_absorbed:
            lines.append(f"Absorbed FE: {', '.join(self.fe_absorbed)} ({self.df_absorbed} DoF)")
        if self.n_clusters:
            for name, g in self.n_clusters.items():
                lines.append(f"Clusters ({name}): {g}")
        lines.append(f"{'-'*60}")
        lines.append(f"{'':>12} {'Coef':>10} {'SE':>10} {'t':>8} {'P>|t|':>8} {'[0.025':>8} {'0.975]':>8}")
        lines.append(f"{'-'*60}")
        ci = self.confint()
        for i, name in enumerate(self.names):
            lines.append(
                f"{name:>12} {self.coefficients[i]:>10.4f} {self.se[i]:>10.4f} "
                f"{self.tstat[i]:>8.2f} {self.pvalue[i]:>8.4f} "
                f"{ci[i, 0]:>8.4f} {ci[i, 1]:>8.4f}"
            )
        lines.append(f"{'='*60}")
        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_results.py -v`

**Step 5: Commit**

```bash
git add polars_reg/_results.py tests/test_results.py
git commit -m "feat: RegressionResult dataclass with summary display"
```

### Task 5.2: OLS estimator

**Files:**
- Create: `polars_reg/_ols.py`
- Create: `tests/test_ols.py`

**Step 1: Write the failing test**

```python
# tests/test_ols.py
import numpy as np
import polars as pl

from polars_reg._ols import ols


def test_ols_basic(simple_data):
    result = ols("y ~ x1 + x2", data=simple_data)
    # DGP: y = 2.0 + 1.5*x1 - 0.5*x2 + N(0, 0.25)
    assert result.n_obs == 1000
    assert result.model_type == "OLS"
    np.testing.assert_allclose(result.coefficients[0], 1.5, atol=0.1)   # x1
    np.testing.assert_allclose(result.coefficients[1], -0.5, atol=0.1)  # x2
    np.testing.assert_allclose(result.coefficients[2], 2.0, atol=0.1)   # _cons
    assert result.r_squared > 0.8


def test_ols_robust(simple_data):
    result = ols("y ~ x1 + x2", data=simple_data, vcov="HC1")
    assert result.vcov_type == "HC1"
    assert result.se is not None
    assert len(result.se) == 3


def test_ols_clustered(panel_data):
    result = ols("y ~ x1 + x2", data=panel_data, cluster=["firm_id"])
    assert result.vcov_type == "cluster"
    assert result.n_clusters == {"firm_id": 50}


def test_ols_twoway_clustered(panel_data):
    result = ols("y ~ x1 + x2", data=panel_data, cluster=["firm_id", "year_id"])
    assert result.vcov_type == "cluster"
    assert result.n_clusters == {"firm_id": 50, "year_id": 20}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ols.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement OLS**

```python
# polars_reg/_ols.py
from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import vcov_clustered, vcov_iid, vcov_multiway_clustered, vcov_robust
from polars_reg._utils import extract_arrays


def ols(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Ordinary Least Squares regression.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        vcov: "iid", "HC0", "HC1", "HC2", or "HC3"
        cluster: Column name(s) for clustered SEs. Overrides vcov.
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster)

    X, y = arrays.X, arrays.y
    n, k = X.shape

    # Solve OLS: beta = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    resid = y - X @ beta

    # R-squared
    ss_res = resid @ resid
    y_demean = y - y.mean()
    ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    # Variance-covariance
    if cluster:
        cluster_arrays = [arrays.cluster_arrays[c] for c in cluster]
        if len(cluster_arrays) == 1:
            V = vcov_clustered(X, resid, cluster_arrays[0])
        else:
            V = vcov_multiway_clustered(X, resid, cluster_arrays)
        vcov_type = "cluster"
        n_clusters = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters.values()) - 1
    elif vcov == "iid":
        V = vcov_iid(X, resid)
        vcov_type = "iid"
        n_clusters = None
        df_r = n - k
    else:
        V = vcov_robust(X, resid, kind=vcov)
        vcov_type = vcov
        n_clusters = None
        df_r = n - k

    return RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=arrays.names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="OLS",
        vcov_type=vcov_type,
        n_clusters=n_clusters,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ols.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add polars_reg/_ols.py tests/test_ols.py
git commit -m "feat: OLS estimator with iid, robust, and clustered SEs"
```

---

## Phase 6: Iterative Demeaning (FE Absorption)

### Task 6.1: Single-FE within-demeaning

**Files:**
- Create: `polars_reg/_demean.py`
- Create: `tests/test_demean.py`

**Step 1: Write the failing test**

```python
# tests/test_demean.py
import numpy as np

from polars_reg._demean import demean


def test_single_fe_demean():
    """Demeaning by one FE should subtract group means."""
    rng = np.random.default_rng(42)
    n = 100
    groups = np.repeat(np.arange(10), 10)
    x = rng.standard_normal(n)

    result = demean(x.reshape(-1, 1), {"g": groups})
    # After demeaning, group means should be ~0
    for g in range(10):
        mask = groups == g
        np.testing.assert_allclose(result[mask, 0].mean(), 0.0, atol=1e-12)


def test_demean_preserves_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 3))
    groups = np.repeat(np.arange(10), 10)
    result = demean(X, {"g": groups})
    assert result.shape == (100, 3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_demean.py -v`

**Step 3: Implement demeaning**

```python
# polars_reg/_demean.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _group_means(
    x: NDArray, codes: NDArray, n_groups: int
) -> NDArray:
    """Compute group means for a 1D or 2D array. Returns array of shape (n_groups,) or (n_groups, k)."""
    if x.ndim == 1:
        sums = np.bincount(codes, weights=x, minlength=n_groups)
        counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
        return sums / counts
    else:
        k = x.shape[1]
        means = np.empty((n_groups, k))
        counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
        for j in range(k):
            sums = np.bincount(codes, weights=x[:, j], minlength=n_groups)
            means[:, j] = sums / counts
        return means


def demean(
    X: NDArray,
    fe_dict: dict[str, NDArray],
    tol: float = 1e-8,
    max_iter: int = 100_000,
) -> NDArray:
    """Demean columns of X by absorbing multiple fixed effects.

    Uses the Symmetric Kaczmarz (Symmetric Halperin) transform with
    conjugate gradient acceleration, following Correia (2016).

    For a single FE, exact demeaning in one pass.
    For multiple FE, iterative alternating projections.

    Args:
        X: (n, k) array of variables to demean
        fe_dict: mapping of FE name -> integer codes array
        tol: convergence tolerance (relative change in L2 norm)
        max_iter: maximum iterations
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    X = X.copy().astype(np.float64)
    fe_list = list(fe_dict.values())
    n_groups_list = [int(codes.max()) + 1 for codes in fe_list]

    if len(fe_list) == 1:
        # Single FE: exact in one pass
        codes = fe_list[0]
        n_g = n_groups_list[0]
        means = _group_means(X, codes, n_g)
        X -= means[codes]
        return X.squeeze() if squeeze else X

    # Multiple FE: Symmetric Kaczmarz + Conjugate Gradient
    result = _demean_cg(X, fe_list, n_groups_list, tol, max_iter)
    return result.squeeze() if squeeze else result


def _symmetric_kaczmarz(
    X: NDArray, fe_list: list[NDArray], n_groups_list: list[int]
) -> NDArray:
    """One sweep of symmetric Kaczmarz: forward then backward through FE dims."""
    # Forward
    for codes, n_g in zip(fe_list, n_groups_list):
        means = _group_means(X, codes, n_g)
        X = X - means[codes]
    # Backward (skip last since forward already did it)
    for codes, n_g in zip(reversed(fe_list[:-1]), reversed(n_groups_list[:-1])):
        means = _group_means(X, codes, n_g)
        X = X - means[codes]
    return X


def _demean_cg(
    X: NDArray,
    fe_list: list[NDArray],
    n_groups_list: list[int],
    tol: float,
    max_iter: int,
) -> NDArray:
    """Conjugate gradient acceleration with symmetric Kaczmarz transform."""
    # The operator A we are solving is: Ax = x - T(x), where T is symmetric Kaczmarz
    # We want to find x* such that T(x*) = x* (fixed point), i.e. Ax* = 0
    # Equivalently, solve Ax = b where b = x_original - T(x_original), starting from x=0
    # But it's simpler to iterate: x_{k+1} = T(x_k) with CG acceleration

    n, k = X.shape
    x = X.copy()
    r = _symmetric_kaczmarz(x.copy(), fe_list, n_groups_list) - x  # residual: T(x) - x
    # Actually: we want x such that x = T(x). Residual = T(x) - x.
    # CG on the normal equations for the fixed point.

    # Simpler CG formulation from pyhdfe:
    # x is our current estimate. r = T(x) - x (residual).
    # We want r -> 0.
    r = _symmetric_kaczmarz(x.copy(), fe_list, n_groups_list) - x
    u = r.copy()
    ssr = np.sum(r**2)

    for iteration in range(max_iter):
        if ssr < tol**2 * max(np.sum(x**2), 1e-16):
            break

        v = u - _symmetric_kaczmarz(u.copy(), fe_list, n_groups_list)  # A*u = u - T(u)
        uv = np.sum(u * v)
        if abs(uv) < 1e-30:
            break
        alpha = ssr / uv
        x = x + alpha * u
        r = r - alpha * v
        ssr_new = np.sum(r**2)
        beta = ssr_new / ssr
        u = r + beta * u
        ssr = ssr_new
    else:
        import warnings
        warnings.warn(f"Demeaning did not converge after {max_iter} iterations")

    return x
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_demean.py -v`

**Step 5: Commit**

```bash
git add polars_reg/_demean.py tests/test_demean.py
git commit -m "feat: iterative demeaning with symmetric Kaczmarz + CG acceleration"
```

### Task 6.2: Multi-way FE demeaning correctness test

**Files:**
- Modify: `tests/test_demean.py`

**Step 1: Write the failing test**

```python
def test_twoway_fe_demean():
    """Two-way demeaning should match brute-force LSDV projection."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 10, 5
    n = n_firms * n_years
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(n_years), n_firms)
    x = rng.standard_normal(n)

    # Brute force: regress x on firm + year dummies, take residuals
    D_firm = np.eye(n_firms)[firm]
    D_year = np.eye(n_years)[year]
    D = np.column_stack([D_firm, D_year])
    proj = D @ np.linalg.lstsq(D, x, rcond=None)[0]
    expected = x - proj

    result = demean(x.reshape(-1, 1), {"firm": firm, "year": year})
    np.testing.assert_allclose(result[:, 0], expected, atol=1e-6)
```

**Step 2: Run test**

Run: `pytest tests/test_demean.py::test_twoway_fe_demean -v`
Expected: PASS (if CG implementation is correct). If FAIL, debug the CG implementation.

**Step 3: Commit**

```bash
git add tests/test_demean.py
git commit -m "test: two-way FE demeaning matches brute-force LSDV projection"
```

### Task 6.3: Singleton detection and connected components DoF

**Files:**
- Modify: `polars_reg/_demean.py`
- Modify: `tests/test_demean.py`

**Step 1: Write the failing test**

```python
from polars_reg._demean import drop_singletons, absorbed_dof


def test_drop_singletons():
    # obs 0 is the only member of group 99 in FE 'a'
    a = np.array([99, 0, 0, 1, 1, 2, 2, 2])
    b = np.array([0, 0, 1, 0, 1, 0, 1, 2])
    mask = drop_singletons({"a": a, "b": b})
    assert mask[0] == False  # singleton in group 99
    assert mask[1:].all()


def test_absorbed_dof_single_fe():
    codes = np.array([0, 0, 1, 1, 2, 2])
    dof = absorbed_dof({"g": codes})
    assert dof == 3  # 3 groups


def test_absorbed_dof_twoway():
    """Two-way FE: dof = g1 + g2 - connected_components."""
    firm = np.array([0, 0, 1, 1])
    year = np.array([0, 1, 0, 1])
    # Fully connected bipartite graph: 1 component
    dof = absorbed_dof({"firm": firm, "year": year})
    assert dof == 2 + 2 - 1  # = 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_demean.py -v`

**Step 3: Implement**

Add to `polars_reg/_demean.py`:

```python
import scipy.sparse
import scipy.sparse.csgraph


def drop_singletons(fe_dict: dict[str, NDArray]) -> NDArray:
    """Return boolean mask of observations to keep (iteratively drop singletons)."""
    n = len(next(iter(fe_dict.values())))
    keep = np.ones(n, dtype=bool)
    changed = True
    while changed:
        changed = False
        for codes in fe_dict.values():
            counts = np.bincount(codes[keep])
            singleton_groups = np.where(counts == 1)[0]
            if len(singleton_groups) > 0:
                for g in singleton_groups:
                    mask = (codes == g) & keep
                    if mask.any():
                        keep[mask] = False
                        changed = True
    return keep


def absorbed_dof(fe_dict: dict[str, NDArray]) -> int:
    """Count degrees of freedom absorbed by fixed effects.

    Single FE: number of groups.
    Two+ FE: sum of groups minus connected components (pairwise method).
    """
    fe_list = list(fe_dict.values())
    n_groups = [int(codes.max()) + 1 for codes in fe_list]
    total_dof = n_groups[0]

    for i in range(1, len(fe_list)):
        total_dof += n_groups[i]
        # Subtract max connected components between dim i and any prior dim
        max_components = 0
        for j in range(i):
            c = _connected_components(fe_list[j], n_groups[j], fe_list[i], n_groups[i])
            max_components = max(max_components, c)
        total_dof -= max_components

    return total_dof


def _connected_components(
    codes_a: NDArray, n_a: int, codes_b: NDArray, n_b: int
) -> int:
    """Count connected components in bipartite graph of two FE dimensions."""
    total = n_a + n_b
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(codes_a)), (codes_a, n_a + codes_b)),
        shape=(total, total),
    )
    graph = graph + graph.T  # make symmetric
    n_components = scipy.sparse.csgraph.connected_components(graph, directed=False)[0]
    return n_components
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_demean.py -v`

**Step 5: Commit**

```bash
git add polars_reg/_demean.py tests/test_demean.py
git commit -m "feat: singleton detection and connected-component DoF counting"
```

---

## Phase 7: OLS with Absorbed Fixed Effects (reghdfe)

### Task 7.1: reghdfe-style estimation

**Files:**
- Modify: `polars_reg/_ols.py`
- Modify: `tests/test_ols.py`

**Step 1: Write the failing test**

```python
def test_ols_with_fe(panel_data):
    """OLS with absorbed FE should recover coefficients without FE bias."""
    result = ols("y ~ x1 + x2 | firm_id + year_id", data=panel_data)
    # DGP: y = 1.0*x1 - 2.0*x2 + firm_fe + year_fe + e
    assert result.fe_absorbed == ["firm_id", "year_id"]
    np.testing.assert_allclose(result.coefficients[0], 1.0, atol=0.1)  # x1
    np.testing.assert_allclose(result.coefficients[1], -2.0, atol=0.1)  # x2
    assert len(result.names) == 2  # no intercept when FE absorbed
    assert result.df_absorbed > 0


def test_ols_fe_clustered(panel_data):
    result = ols("y ~ x1 + x2 | firm_id + year_id", data=panel_data, cluster=["firm_id"])
    assert result.vcov_type == "cluster"
    assert result.n_clusters == {"firm_id": 50}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ols.py::test_ols_with_fe -v`

**Step 3: Modify `_ols.py` to handle FE absorption**

Update the `ols()` function in `polars_reg/_ols.py` to add FE logic:

```python
# Add import at top
from polars_reg._demean import absorbed_dof, demean, drop_singletons

# In ols(), after extracting arrays and before solving:
def ols(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster)

    X, y = arrays.X, arrays.y
    fe_dict = arrays.fe_arrays

    # Handle fixed effects
    has_fe = len(fe_dict) > 0
    if has_fe:
        # Drop singletons
        keep = drop_singletons(fe_dict)
        if not keep.all():
            y = y[keep]
            X = X[keep]
            fe_dict = {k: v[keep] for k, v in fe_dict.items()}
            if cluster:
                arrays.cluster_arrays = {k: v[keep] for k, v in arrays.cluster_arrays.items()}

        # Remove intercept (absorbed by FE)
        if spec.add_intercept and arrays.names[-1] == "_cons":
            X = X[:, :-1]
            arrays.names = arrays.names[:-1]

        # Demean y and X
        all_vars = np.column_stack([y.reshape(-1, 1), X])
        demeaned = demean(all_vars, fe_dict)
        y = demeaned[:, 0]
        X = demeaned[:, 1:]

        df_abs = absorbed_dof(fe_dict)
    else:
        df_abs = 0

    n, k = X.shape

    # Solve OLS
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    resid = y - X @ beta

    # R-squared (within for FE models)
    ss_res = resid @ resid
    y_demean = y - y.mean()
    ss_tot = y_demean @ y_demean
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - df_abs)

    # Variance-covariance (use demeaned X for sandwich)
    if cluster:
        cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
        if len(cluster_arrays_list) == 1:
            V = vcov_clustered(X, resid, cluster_arrays_list[0])
        else:
            V = vcov_multiway_clustered(X, resid, cluster_arrays_list)
        vcov_type = "cluster"
        n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters_dict.values()) - 1
    elif vcov == "iid":
        V = vcov_iid(X, resid)
        vcov_type = "iid"
        n_clusters_dict = None
        df_r = n - k - df_abs
    else:
        V = vcov_robust(X, resid, kind=vcov)
        vcov_type = vcov
        n_clusters_dict = None
        df_r = n - k - df_abs

    return RegressionResult(
        coefficients=beta,
        vcov=V,
        residuals=resid,
        names=arrays.names,
        n_obs=n,
        k=k,
        df_r=df_r,
        r_squared=r2,
        r_squared_adj=r2_adj,
        model_type="OLS",
        vcov_type=vcov_type,
        n_clusters=n_clusters_dict,
        fe_absorbed=list(fe_dict.keys()) if has_fe else None,
        df_absorbed=df_abs,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ols.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add polars_reg/_ols.py tests/test_ols.py
git commit -m "feat: OLS with absorbed multi-way fixed effects (reghdfe-style)"
```

---

## Phase 8: IV / 2SLS Estimator

### Task 8.1: Two-stage least squares

**Files:**
- Create: `polars_reg/_iv.py`
- Create: `tests/test_iv.py`

**Step 1: Write the failing test**

```python
# tests/test_iv.py
import numpy as np

from polars_reg._iv import iv2sls


def test_iv2sls_basic(iv_data):
    """2SLS should correct endogeneity bias."""
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    # DGP: y = 1.0 + 2.0*x_endog + 0.5*x_exog + u, corr(x_endog, u) > 0
    assert result.model_type == "2SLS"
    np.testing.assert_allclose(result.coefficients[0], 0.5, atol=0.3)   # x_exog
    np.testing.assert_allclose(result.coefficients[1], 2.0, atol=0.3)   # x_endog
    assert result.n_obs == 1000


def test_iv2sls_ols_bias(iv_data):
    """OLS on endogenous model should give biased coefficients."""
    from polars_reg._ols import ols
    ols_result = ols("y ~ x_exog + x_endog", data=iv_data)
    iv_result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    # OLS coefficient on x_endog should be biased upward (positive corr with u)
    ols_endog_idx = ols_result.names.index("x_endog")
    iv_endog_idx = iv_result.names.index("x_endog")
    assert ols_result.coefficients[ols_endog_idx] > iv_result.coefficients[iv_endog_idx]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_iv.py -v`

**Step 3: Implement 2SLS**

```python
# polars_reg/_iv.py
from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import vcov_clustered, vcov_multiway_clustered
from polars_reg._utils import extract_arrays


def iv2sls(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Two-stage least squares (2SLS) IV regression.

    Formula: y ~ exog_vars || endog_vars ~ instruments
    Or with FE: y ~ exog_vars | fe_vars | endog_vars ~ instruments
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster)

    y = arrays.y
    X_exog = arrays.X  # includes intercept if specified
    X_endog = arrays.endog
    Z_excluded = arrays.instruments

    assert X_endog is not None, "No endogenous variables specified"
    assert Z_excluded is not None, "No instruments specified"

    n = len(y)

    # Full instrument matrix: [exogenous regressors, excluded instruments]
    Z = np.column_stack([X_exog, Z_excluded])

    # Stage 1: regress each endogenous var on all instruments
    Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)  # projection matrix
    X_endog_hat = Pz @ X_endog

    # Stage 2: regress y on exog + fitted endogenous
    X_full = np.column_stack([X_exog, X_endog])
    X_hat = np.column_stack([X_exog, X_endog_hat])

    # 2SLS estimator: beta = (X_hat' X)^{-1} X_hat' y
    beta = np.linalg.solve(X_hat.T @ X_full, X_hat.T @ y)

    # Use original X (not X_hat) for residuals
    resid = y - X_full @ beta
    k = X_full.shape[1]

    names = arrays.names + (arrays.endog_names or [])

    # R-squared
    ss_res = resid @ resid
    ss_tot = (y - y.mean()) @ (y - y.mean())
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    # Standard errors: use X_hat for bread, original resid for meat
    XhXi = np.linalg.inv(X_hat.T @ X_full)

    if cluster:
        cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
        # Clustered sandwich with X_hat
        score = X_hat * resid[:, None]
        meat = np.zeros((k, k))
        cl = cluster_arrays_list[0] if len(cluster_arrays_list) == 1 else cluster_arrays_list
        # For simplicity, use the standard clustered formula with X_hat
        if len(cluster_arrays_list) == 1:
            V = _iv_vcov_clustered(X_hat, X_full, resid, cluster_arrays_list[0])
        else:
            V = _iv_vcov_multiway(X_hat, X_full, resid, cluster_arrays_list)
        vcov_type = "cluster"
        n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}
        df_r = min(n_clusters_dict.values()) - 1
    elif vcov == "iid":
        sigma2 = resid @ resid / (n - k)
        V = sigma2 * np.linalg.inv(X_hat.T @ X_full)
        vcov_type = "iid"
        n_clusters_dict = None
        df_r = n - k
    else:
        # Robust: sandwich with X_hat as bread, heteroskedastic meat
        meat = X_hat.T @ (X_hat * (resid**2)[:, None])
        V = (n / (n - k)) * XhXi @ meat @ XhXi.T
        vcov_type = vcov
        n_clusters_dict = None
        df_r = n - k

    return RegressionResult(
        coefficients=beta, vcov=V, residuals=resid, names=names,
        n_obs=n, k=k, df_r=df_r, r_squared=r2, r_squared_adj=r2_adj,
        model_type="2SLS", vcov_type=vcov_type, n_clusters=n_clusters_dict,
    )


def _iv_vcov_clustered(
    X_hat: np.ndarray, X: np.ndarray, resid: np.ndarray, clusters: np.ndarray
) -> np.ndarray:
    """Clustered VCV for IV: bread uses X_hat, meat uses clustered scores."""
    n, k = X_hat.shape
    XhX_inv = np.linalg.inv(X_hat.T @ X)
    score = X_hat * resid[:, None]
    unique = np.unique(clusters)
    G = len(unique)
    meat = np.zeros((k, k))
    for g in unique:
        mask = clusters == g
        sg = score[mask].sum(axis=0)
        meat += np.outer(sg, sg)
    dfc = (G / (G - 1)) * ((n - 1) / (n - k))
    return dfc * XhX_inv @ meat @ XhX_inv.T


def _iv_vcov_multiway(
    X_hat: np.ndarray, X: np.ndarray, resid: np.ndarray,
    cluster_list: list[np.ndarray],
) -> np.ndarray:
    """Multi-way clustered VCV for IV."""
    from itertools import combinations

    from polars_reg._se import _interaction_codes

    D = len(cluster_list)
    n, k = X_hat.shape
    XhX_inv = np.linalg.inv(X_hat.T @ X)
    score = X_hat * resid[:, None]
    V = np.zeros((k, k))

    for size in range(1, D + 1):
        sign = (-1) ** (size + 1)
        for subset in combinations(range(D), size):
            subset_arrays = [cluster_list[d] for d in subset]
            interaction = _interaction_codes(*subset_arrays)
            unique = np.unique(interaction)
            G = len(unique)
            meat = np.zeros((k, k))
            for g in unique:
                mask = interaction == g
                sg = score[mask].sum(axis=0)
                meat += np.outer(sg, sg)
            dfc = (G / (G - 1)) * ((n - 1) / (n - k))
            V += sign * dfc * XhX_inv @ meat @ XhX_inv.T

    return V
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_iv.py -v`

**Step 5: Commit**

```bash
git add polars_reg/_iv.py tests/test_iv.py
git commit -m "feat: 2SLS IV estimator with robust and clustered SEs"
```

### Task 8.2: First-stage diagnostics (F-stat, weak instrument tests)

**Files:**
- Modify: `polars_reg/_iv.py`
- Modify: `tests/test_iv.py`

**Step 1: Write the failing test**

```python
def test_first_stage_f_stat(iv_data):
    result = iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.first_stage_f is not None
    assert result.first_stage_f > 10  # instruments are relevant in DGP
```

**Step 2: Implement first-stage F-stat**

Add `first_stage_f` to `RegressionResult` as an optional field, compute it in `iv2sls` as the joint F-stat from the first-stage regression of each endogenous variable on all instruments (partial F-test excluding exogenous regressors).

```python
# In iv2sls(), after stage 1:
# First-stage F-stat (Sanderson-Windmeijer for each endogenous var)
# Simple case: single endogenous var
if X_endog.ndim == 1 or X_endog.shape[1] == 1:
    x_end = X_endog.ravel()
    # Partial F: compare R² of Z on x_endog vs X_exog on x_endog
    resid_reduced = x_end - X_exog @ np.linalg.solve(X_exog.T @ X_exog, X_exog.T @ x_end)
    resid_full = x_end - Z @ np.linalg.solve(Z.T @ Z, Z.T @ x_end)
    q = Z_excluded.shape[1]  # number of excluded instruments
    ss_reduced = resid_reduced @ resid_reduced
    ss_full = resid_full @ resid_full
    first_stage_f = ((ss_reduced - ss_full) / q) / (ss_full / (n - Z.shape[1]))
else:
    first_stage_f = None  # TODO: Sanderson-Windmeijer for multiple endog
```

**Step 3: Run tests, commit**

---

## Phase 9: LIML Estimator

### Task 9.1: LIML estimation

**Files:**
- Modify: `polars_reg/_iv.py`
- Modify: `tests/test_iv.py`

**Step 1: Write the failing test**

```python
from polars_reg._iv import liml


def test_liml_basic(iv_data):
    result = liml("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.model_type == "LIML"
    np.testing.assert_allclose(result.coefficients[1], 2.0, atol=0.3)
```

**Step 2: Implement LIML**

LIML finds the minimum eigenvalue kappa of (Y'Mz Y)^{-1} (Y'Mx Y), where Y = [y, X_endog], Mz is the residual maker from projecting off Z (all instruments), Mx is the residual maker from projecting off X_exog only. Then beta_LIML uses weight kappa in a modified 2SLS formula.

```python
def liml(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Limited Information Maximum Likelihood IV estimator."""
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster)

    y = arrays.y
    X_exog = arrays.X
    X_endog = arrays.endog
    Z_excluded = arrays.instruments
    n = len(y)

    Z = np.column_stack([X_exog, Z_excluded])
    Y = np.column_stack([y, X_endog])

    # Residual makers
    Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    Px = X_exog @ np.linalg.solve(X_exog.T @ X_exog, X_exog.T)
    Mz = np.eye(n) - Pz
    Mx = np.eye(n) - Px

    # Find kappa: minimum eigenvalue of (Y'Mz Y)^{-1} (Y'Mx Y)
    A = Y.T @ Mz @ Y
    B = Y.T @ Mx @ Y
    eigvals = np.linalg.eigvalsh(np.linalg.solve(A, B))
    kappa = eigvals.min()

    # LIML estimator: (X'(I - kappa*Mz)X)^{-1} X'(I - kappa*Mz)y
    # Equivalently: weight matrix W = I - kappa*Mz + kappa*Pz = (1-kappa)*I + kappa*Pz
    # But simpler: X_w = X - kappa*(X - Pz@X), y_w = y - kappa*(y - Pz@y)
    X_full = np.column_stack([X_exog, X_endog])
    W = np.eye(n) - kappa * Mz  # = (1-kappa)*I + kappa*Pz
    X_w = W @ X_full
    y_w = W @ y

    beta = np.linalg.solve(X_w.T @ X_full, X_w.T @ y)
    resid = y - X_full @ beta
    k = X_full.shape[1]

    names = arrays.names + (arrays.endog_names or [])
    ss_res = resid @ resid
    ss_tot = (y - y.mean()) @ (y - y.mean())
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    # Standard errors (same sandwich structure as 2SLS but with W-weighted bread)
    if vcov == "iid":
        sigma2 = resid @ resid / (n - k)
        V = sigma2 * np.linalg.inv(X_w.T @ X_full)
    else:
        XwX_inv = np.linalg.inv(X_w.T @ X_full)
        meat = X_w.T @ (X_w * (resid**2)[:, None])
        V = (n / (n - k)) * XwX_inv @ meat @ XwX_inv.T

    return RegressionResult(
        coefficients=beta, vcov=V, residuals=resid, names=names,
        n_obs=n, k=k, df_r=n - k, r_squared=r2, r_squared_adj=r2_adj,
        model_type="LIML", vcov_type=vcov,
    )
```

**Step 3: Run tests, commit**

```bash
git add polars_reg/_iv.py tests/test_iv.py
git commit -m "feat: LIML IV estimator"
```

---

## Phase 10: GMM Estimator

### Task 10.1: Two-step efficient GMM

**Files:**
- Create: `polars_reg/_gmm.py`
- Create: `tests/test_gmm.py`

**Step 1: Write the failing test**

```python
# tests/test_gmm.py
import numpy as np
from polars_reg._gmm import gmm_iv


def test_gmm_basic(iv_data):
    result = gmm_iv("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.model_type == "GMM"
    np.testing.assert_allclose(result.coefficients[1], 2.0, atol=0.3)


def test_gmm_overid_test(iv_data):
    """With 2 instruments and 1 endogenous var, Hansen J should be available."""
    result = gmm_iv("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data)
    assert result.j_stat is not None
    assert result.j_pvalue is not None
```

**Step 2: Implement two-step GMM**

```python
# polars_reg/_gmm.py
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from polars_reg._formula import parse_formula
from polars_reg._results import RegressionResult
from polars_reg._utils import extract_arrays


def gmm_iv(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "robust",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Two-step efficient GMM-IV estimator.

    Step 1: 2SLS (identity weight matrix).
    Step 2: Re-weight using optimal weight matrix from step-1 residuals.
    """
    if isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster)

    y, X_exog, X_endog, Z_excl = arrays.y, arrays.X, arrays.endog, arrays.instruments
    n = len(y)
    Z = np.column_stack([X_exog, Z_excl])
    X = np.column_stack([X_exog, X_endog])
    k = X.shape[1]
    names = arrays.names + (arrays.endog_names or [])

    # Step 1: 2SLS (W = (Z'Z)^{-1})
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    beta_1 = np.linalg.solve(X.T @ Z @ ZtZ_inv @ Z.T @ X, X.T @ Z @ ZtZ_inv @ Z.T @ y)
    resid_1 = y - X @ beta_1

    # Step 2: Optimal weight matrix
    # Robust: W_opt = (1/n * Z' diag(e1^2) Z)^{-1}
    S = Z.T @ (Z * (resid_1**2)[:, None]) / n
    S_inv = np.linalg.inv(S)
    beta_2 = np.linalg.solve(X.T @ Z @ S_inv @ Z.T @ X, X.T @ Z @ S_inv @ Z.T @ y)
    resid = y - X @ beta_2

    # VCV: (X'Z W Z'X)^{-1}
    XZ = X.T @ Z
    bread = np.linalg.inv(XZ @ S_inv @ XZ.T)
    V = bread / n

    # Hansen J-test (overidentification)
    q = Z_excl.shape[1]
    n_endog = X_endog.shape[1] if X_endog.ndim > 1 else 1
    overid_dof = q - n_endog
    j_stat = None
    j_pvalue = None
    if overid_dof > 0:
        g_bar = Z.T @ resid / n
        S_final = Z.T @ (Z * (resid**2)[:, None]) / n
        S_final_inv = np.linalg.inv(S_final)
        j_stat = float(n * g_bar @ S_final_inv @ g_bar)
        j_pvalue = float(1.0 - stats.chi2.cdf(j_stat, overid_dof))

    ss_res = resid @ resid
    ss_tot = (y - y.mean()) @ (y - y.mean())
    r2 = 1.0 - ss_res / ss_tot

    result = RegressionResult(
        coefficients=beta_2, vcov=V, residuals=resid, names=names,
        n_obs=n, k=k, df_r=n - k, r_squared=r2,
        r_squared_adj=1.0 - (1.0 - r2) * (n - 1) / (n - k),
        model_type="GMM", vcov_type="robust",
    )
    result.j_stat = j_stat
    result.j_pvalue = j_pvalue
    return result
```

Note: Add `j_stat` and `j_pvalue` as optional fields on `RegressionResult`.

**Step 3: Run tests, commit**

```bash
git add polars_reg/_gmm.py tests/test_gmm.py polars_reg/_results.py
git commit -m "feat: two-step efficient GMM-IV with Hansen J overidentification test"
```

---

## Phase 11: Panel Estimators

### Task 11.1: Panel fixed effects (within estimator)

**Files:**
- Create: `polars_reg/_panel.py`
- Create: `tests/test_panel.py`

**Step 1: Write the failing test**

```python
# tests/test_panel.py
import numpy as np
from polars_reg._panel import panel_fe


def test_panel_fe(panel_data):
    """Panel FE (within estimator) should match OLS with absorbed entity FE."""
    result = panel_fe("y ~ x1 + x2", data=panel_data, entity="firm_id", time="year_id")
    assert result.model_type == "Panel FE"
    np.testing.assert_allclose(result.coefficients[0], 1.0, atol=0.1)
    np.testing.assert_allclose(result.coefficients[1], -2.0, atol=0.1)
```

**Step 2: Implement**

Panel FE is a thin wrapper over the demeaning infrastructure — demean by entity, run OLS on demeaned data. Same as `ols("y ~ x1 + x2 | firm_id", data)` but with panel-specific DoF and diagnostics.

```python
# polars_reg/_panel.py
from __future__ import annotations

import numpy as np
import polars as pl

from polars_reg._demean import absorbed_dof, demean, drop_singletons
from polars_reg._formula import FormulaSpec, parse_formula
from polars_reg._results import RegressionResult
from polars_reg._se import vcov_clustered, vcov_iid, vcov_robust
from polars_reg._utils import extract_arrays


def panel_fe(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str | None = None,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
) -> RegressionResult:
    """Panel fixed effects (within) estimator.

    Demeans by entity (and optionally time), then OLS on demeaned data.
    Default clusters SEs by entity.
    """
    if cluster is None:
        cluster = [entity]
    elif isinstance(cluster, str):
        cluster = [cluster]

    spec = parse_formula(formula)
    # Add entity (and time) as FE
    spec.fe = [entity] + ([time] if time else [])
    spec.add_intercept = False

    arrays = extract_arrays(data, spec, cluster=cluster)
    y, X = arrays.y, arrays.X
    fe_dict = arrays.fe_arrays

    # Remove intercept column if present
    if arrays.names and arrays.names[-1] == "_cons":
        X = X[:, :-1]
        arrays.names = arrays.names[:-1]

    keep = drop_singletons(fe_dict)
    if not keep.all():
        y, X = y[keep], X[keep]
        fe_dict = {k: v[keep] for k, v in fe_dict.items()}
        arrays.cluster_arrays = {k: v[keep] for k, v in arrays.cluster_arrays.items()}

    all_vars = np.column_stack([y.reshape(-1, 1), X])
    demeaned = demean(all_vars, fe_dict)
    y_dm, X_dm = demeaned[:, 0], demeaned[:, 1:]

    n, k = X_dm.shape
    df_abs = absorbed_dof(fe_dict)

    beta = np.linalg.solve(X_dm.T @ X_dm, X_dm.T @ y_dm)
    resid = y_dm - X_dm @ beta

    ss_res = resid @ resid
    ss_tot = (y_dm - y_dm.mean()) @ (y_dm - y_dm.mean())
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - df_abs)

    cluster_arrays_list = [arrays.cluster_arrays[c] for c in cluster]
    if len(cluster_arrays_list) == 1:
        V = vcov_clustered(X_dm, resid, cluster_arrays_list[0])
    else:
        from polars_reg._se import vcov_multiway_clustered
        V = vcov_multiway_clustered(X_dm, resid, cluster_arrays_list)
    n_clusters_dict = {c: len(np.unique(arrays.cluster_arrays[c])) for c in cluster}

    return RegressionResult(
        coefficients=beta, vcov=V, residuals=resid, names=arrays.names,
        n_obs=n, k=k, df_r=min(n_clusters_dict.values()) - 1,
        r_squared=r2, r_squared_adj=r2_adj,
        model_type="Panel FE", vcov_type="cluster",
        n_clusters=n_clusters_dict,
        fe_absorbed=list(fe_dict.keys()), df_absorbed=df_abs,
    )
```

**Step 3: Run tests, commit**

### Task 11.2: Random effects (GLS) and first-difference — follow same pattern

These are lower priority. Random effects requires estimating theta from between/within variance ratio and transforming variables as `y_it - theta * y_bar_i`. First-difference transforms by `y_it - y_{i,t-1}`. Both are straightforward extensions of the demeaning/transformation infrastructure.

---

## Phase 12: Public API

### Task 12.1: Wire up __init__.py

**Files:**
- Modify: `polars_reg/__init__.py`

```python
"""polars_reg: Econometric regression methods using Polars DataFrames."""

from polars_reg._gmm import gmm_iv
from polars_reg._iv import iv2sls, liml
from polars_reg._ols import ols
from polars_reg._panel import panel_fe
from polars_reg._results import RegressionResult

__all__ = ["ols", "iv2sls", "liml", "gmm_iv", "panel_fe", "RegressionResult"]
```

### Task 12.2: Integration test with Stata ground truth

**Files:**
- Create: `tests/test_integration.py`

```python
"""Integration tests comparing against known Stata output.

Stata commands and results should be documented inline.
Run in Stata first, record coefficients and SEs to 6+ decimal places.
"""
import numpy as np
import polars as pl
import pytest

from polars_reg import ols


def test_ols_matches_stata():
    """Compare OLS output to Stata: reg y x1 x2, robust"""
    # TODO: Generate a fixed dataset, run in Stata, record exact coefficients/SEs
    pass


def test_reghdfe_matches_stata():
    """Compare to Stata: reghdfe y x1 x2, absorb(firm_id year_id) cluster(firm_id)"""
    # TODO: same approach
    pass


def test_iv_matches_stata():
    """Compare to Stata: ivregress 2sls y x_exog (x_endog = z1 z2), robust"""
    pass
```

**Commit:**

```bash
git add polars_reg/__init__.py tests/test_integration.py
git commit -m "feat: public API and integration test stubs for Stata parity"
```

---

## Dependency Graph

```
Phase 1 (scaffold)
  └─> Phase 2 (formula parser)
  └─> Phase 3 (utils)
        └─> Phase 4 (standard errors)
        └─> Phase 6 (demeaning)
              └─> Phase 7 (OLS + FE = reghdfe)
              └─> Phase 11 (panel)
        └─> Phase 5 (results + OLS)
              └─> Phase 8 (2SLS)
                    └─> Phase 9 (LIML)
              └─> Phase 10 (GMM)
  └─> Phase 12 (public API + integration)
```

Phases 2, 3 can run in parallel. Within Phase 4, tasks are sequential. Phases 8, 9, 10 depend on 5 but are independent of each other. Phase 7 and 11 depend on Phase 6.
