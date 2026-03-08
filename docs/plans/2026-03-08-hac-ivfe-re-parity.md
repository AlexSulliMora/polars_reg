# HAC/DK SEs, IV+FE Tests, RE GLS, Stata Parity — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add HAC/Driscoll-Kraay SEs to all IV estimators with Rust acceleration, add IV+FE test coverage, extend panel RE with full SE support, and validate everything against frozen Stata fixtures.

**Architecture:** Factor HAC/DK meat computation out of existing `_se.py` functions into standalone helpers (Python + Rust). Extend `iv2sls()`, `liml()`, `gmm_iv()` with `time`/`bandwidth` parameters and NW/DK routing. Extend `panel_re()` with all SE types. Generate Stata fixtures once locally, commit CSVs, test against them in CI.

**Tech Stack:** PyO3, numpy crate, rayon, scipy, Stata (local only for fixture generation)

---

## Task 1: Factor out HAC/DK meat helpers in `_se.py`

Currently `vcov_hac()` and `vcov_driscoll_kraay()` compute bread + meat + sandwich internally. IV estimators need different bread matrices, so we need standalone meat functions.

**Files:**
- Modify: `polars_reg/_se.py`
- Test: `tests/test_se.py`

**Step 1:** Add `_hac_meat()` and `_dk_meat()` helper functions to `_se.py`, right above `vcov_hac()` (before line 135):

```python
def _hac_meat(
    score: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
) -> NDArray:
    """Compute Newey-West HAC meat matrix from score vectors.

    Args:
        score: n x k score matrix (typically X * resid[:, None]).
        time_ids: Time period identifiers. Scores are sorted by these.
        bandwidth: Number of lags. Default: floor(4*(n/100)^(2/9)).

    Returns:
        k x k meat matrix.
    """
    if _HAS_NATIVE:
        try:
            from polars_reg._native import rust_hac_meat as _rust_hac_meat
            return np.asarray(
                _rust_hac_meat(
                    np.ascontiguousarray(score, dtype=np.float64),
                    np.ascontiguousarray(time_ids, dtype=np.float64),
                    bandwidth if bandwidth is not None else -1,
                )
            )
        except ImportError:
            pass

    n = score.shape[0]
    order = np.argsort(time_ids)
    score = score[order]

    if bandwidth is None:
        bandwidth = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))

    S = score.T @ score
    for j in range(1, bandwidth + 1):
        w = 1.0 - j / (bandwidth + 1)
        Gamma_j = score[j:].T @ score[:-j]
        S += w * (Gamma_j + Gamma_j.T)
    return S


def _dk_meat(
    score: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
) -> NDArray:
    """Compute Driscoll-Kraay meat matrix from score vectors.

    Aggregates scores by time period, then applies Newey-West kernel.

    Args:
        score: n x k score matrix (typically X * resid[:, None]).
        time_ids: Time period identifiers.
        bandwidth: Number of lags. Default: floor(4*(T/100)^(2/9)).

    Returns:
        k x k meat matrix, and T (number of time periods) for dfc.
    """
    if _HAS_NATIVE:
        try:
            from polars_reg._native import rust_dk_meat as _rust_dk_meat
            return np.asarray(
                _rust_dk_meat(
                    np.ascontiguousarray(score, dtype=np.float64),
                    np.ascontiguousarray(time_ids, dtype=np.float64),
                    bandwidth if bandwidth is not None else -1,
                )
            )
        except ImportError:
            pass

    k = score.shape[1]
    unique_times, time_idx = np.unique(time_ids, return_inverse=True)
    T = len(unique_times)

    h = np.zeros((T, k))
    for j in range(k):
        h[:, j] = np.bincount(time_idx, weights=score[:, j], minlength=T)

    if bandwidth is None:
        bandwidth = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))

    S = h.T @ h
    for j in range(1, bandwidth + 1):
        w = 1.0 - j / (bandwidth + 1)
        Gamma_j = h[j:].T @ h[:-j]
        S += w * (Gamma_j + Gamma_j.T)
    return S
```

**Step 2:** Refactor `vcov_hac()` (lines 135-168) to use `_hac_meat()`:

```python
def vcov_hac(
    X: NDArray,
    resid: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
) -> NDArray:
    """Newey-West HAC VCV (heteroskedasticity and autocorrelation consistent)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    score = X * resid[:, None]
    S = _hac_meat(score, time_ids, bandwidth)
    dfc = n / (n - k)
    return dfc * XtX_inv @ S @ XtX_inv
```

**Step 3:** Refactor `vcov_driscoll_kraay()` (lines 171-210) to use `_dk_meat()`:

```python
def vcov_driscoll_kraay(
    X: NDArray,
    resid: NDArray,
    time_ids: NDArray,
    bandwidth: int | None = None,
) -> NDArray:
    """Driscoll-Kraay VCV for panel data."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    score = X * resid[:, None]
    S = _dk_meat(score, time_ids, bandwidth)
    T = len(np.unique(time_ids))
    dfc = T / (T - 1)
    return dfc * XtX_inv @ S @ XtX_inv
```

**Step 4:** Write regression test to verify refactoring didn't change outputs:

```python
# tests/test_se.py — add at the end
def test_hac_refactor_parity():
    """Refactored vcov_hac should give identical results to old implementation."""
    rng = np.random.default_rng(42)
    n, k = 500, 3
    X = rng.standard_normal((n, k))
    resid = rng.standard_normal(n)
    time_ids = np.repeat(np.arange(50), 10).astype(float)
    V = vcov_hac(X, resid, time_ids, bandwidth=5)
    # Verify it produces a k x k positive semi-definite matrix
    assert V.shape == (k, k)
    eigvals = np.linalg.eigvalsh(V)
    assert np.all(eigvals >= -1e-10)

def test_dk_refactor_parity():
    """Refactored vcov_driscoll_kraay should give identical results."""
    rng = np.random.default_rng(42)
    n, k = 500, 3
    X = rng.standard_normal((n, k))
    resid = rng.standard_normal(n)
    time_ids = np.repeat(np.arange(50), 10).astype(float)
    V = vcov_driscoll_kraay(X, resid, time_ids, bandwidth=5)
    assert V.shape == (k, k)
    eigvals = np.linalg.eigvalsh(V)
    assert np.all(eigvals >= -1e-10)

def test_hac_meat_standalone():
    """_hac_meat should produce same meat as the full vcov_hac minus bread."""
    rng = np.random.default_rng(42)
    n, k = 500, 3
    X = rng.standard_normal((n, k))
    resid = rng.standard_normal(n)
    time_ids = np.repeat(np.arange(50), 10).astype(float)
    score = X * resid[:, None]
    meat = _hac_meat(score, time_ids, bandwidth=5)
    # Reconstruct full VCV manually
    XtX_inv = np.linalg.inv(X.T @ X)
    dfc = n / (n - k)
    V_manual = dfc * XtX_inv @ meat @ XtX_inv
    V_func = vcov_hac(X, resid, time_ids, bandwidth=5)
    np.testing.assert_allclose(V_manual, V_func, rtol=1e-12)
```

**Step 5:** Run tests:
```bash
pytest tests/test_se.py -v
pytest tests/test_ols.py -v -k "nw or dk or hac"
```

**Step 6:** Commit:
```bash
git add polars_reg/_se.py tests/test_se.py
git commit -m "refactor: factor out _hac_meat/_dk_meat helpers from vcov functions"
```

---

## Task 2: Add Rust HAC/DK meat functions

**Files:**
- Modify: `src/lib.rs`
- Modify: `polars_reg/_se.py` (already has Rust dispatch from Task 1)

**Step 1:** Add `rust_hac_meat` to `src/lib.rs` (before the `#[pymodule]` block at line ~1980):

```rust
/// Newey-West HAC meat matrix: Γ₀ + Σⱼ w(j)(Γⱼ + Γⱼ')
/// score is n×k row-major, time_ids is n×1 (used for sorting).
/// If bandwidth < 0, uses default: floor(4*(n/100)^(2/9)).
#[pyfunction]
fn rust_hac_meat<'py>(
    py: Python<'py>,
    score: PyReadonlyArray2<'py, f64>,
    time_ids: PyReadonlyArray1<'py, f64>,
    bandwidth: i64,
) -> Bound<'py, PyArray2<f64>> {
    let score_arr = score.as_array();
    let time_arr = time_ids.as_array();
    let n = score_arr.nrows();
    let k = score_arr.ncols();

    // Sort by time
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| time_arr[a].partial_cmp(&time_arr[b]).unwrap());

    // Build sorted score matrix (row-major flat)
    let mut s_flat = vec![0.0_f64; n * k];
    for (new_i, &old_i) in order.iter().enumerate() {
        for j in 0..k {
            s_flat[new_i * k + j] = score_arr[[old_i, j]];
        }
    }

    let bw = if bandwidth < 0 {
        std::cmp::max(1, (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize)
    } else {
        bandwidth as usize
    };

    // Γ₀ = S'S
    let mut meat = vec![0.0_f64; k * k];
    for i in 0..n {
        let row = &s_flat[i * k..(i + 1) * k];
        for j in 0..k {
            for l in j..k {
                meat[j * k + l] += row[j] * row[l];
            }
        }
    }

    // Γⱼ for j = 1..bw with Bartlett weights
    for lag in 1..=bw {
        let w = 1.0 - lag as f64 / (bw as f64 + 1.0);
        let mut gamma = vec![0.0_f64; k * k];
        for i in lag..n {
            let row_cur = &s_flat[i * k..(i + 1) * k];
            let row_lag = &s_flat[(i - lag) * k..(i - lag + 1) * k];
            for j in 0..k {
                for l in 0..k {
                    gamma[j * k + l] += row_cur[j] * row_lag[l];
                }
            }
        }
        // Add w * (Γⱼ + Γⱼ') to meat
        for j in 0..k {
            for l in j..k {
                meat[j * k + l] += w * (gamma[j * k + l] + gamma[l * k + j]);
            }
        }
    }

    // Symmetrize
    for j in 0..k {
        for l in (j + 1)..k {
            meat[l * k + j] = meat[j * k + l];
        }
    }

    let result = Array2::from_shape_vec((k, k), meat).unwrap();
    result.into_pyarray(py)
}


/// Driscoll-Kraay meat: aggregate scores by time, then Newey-West on T×k.
#[pyfunction]
fn rust_dk_meat<'py>(
    py: Python<'py>,
    score: PyReadonlyArray2<'py, f64>,
    time_ids: PyReadonlyArray1<'py, f64>,
    bandwidth: i64,
) -> Bound<'py, PyArray2<f64>> {
    let score_arr = score.as_array();
    let time_arr = time_ids.as_array();
    let n = score_arr.nrows();
    let k = score_arr.ncols();

    // Recode time_ids to contiguous 0..T-1
    let mut time_map: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    for i in 0..n {
        let key = time_arr[i].to_bits() as i64;
        let next_id = time_map.len();
        time_map.entry(key).or_insert(next_id);
    }
    let t_count = time_map.len();

    // Aggregate scores by time: h[t, j] = Σ score[i, j] for time[i] == t
    let mut h_flat = vec![0.0_f64; t_count * k];
    for i in 0..n {
        let key = time_arr[i].to_bits() as i64;
        let t = time_map[&key];
        for j in 0..k {
            h_flat[t * k + j] += score_arr[[i, j]];
        }
    }

    let bw = if bandwidth < 0 {
        std::cmp::max(1, (4.0 * (t_count as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize)
    } else {
        bandwidth as usize
    };

    // Γ₀ = h'h
    let mut meat = vec![0.0_f64; k * k];
    for t in 0..t_count {
        let row = &h_flat[t * k..(t + 1) * k];
        for j in 0..k {
            for l in j..k {
                meat[j * k + l] += row[j] * row[l];
            }
        }
    }

    // Bartlett kernel lags
    for lag in 1..=bw {
        let w = 1.0 - lag as f64 / (bw as f64 + 1.0);
        let mut gamma = vec![0.0_f64; k * k];
        for t in lag..t_count {
            let row_cur = &h_flat[t * k..(t + 1) * k];
            let row_lag = &h_flat[(t - lag) * k..(t - lag + 1) * k];
            for j in 0..k {
                for l in 0..k {
                    gamma[j * k + l] += row_cur[j] * row_lag[l];
                }
            }
        }
        for j in 0..k {
            for l in j..k {
                meat[j * k + l] += w * (gamma[j * k + l] + gamma[l * k + j]);
            }
        }
    }

    for j in 0..k {
        for l in (j + 1)..k {
            meat[l * k + j] = meat[j * k + l];
        }
    }

    let result = Array2::from_shape_vec((k, k), meat).unwrap();
    result.into_pyarray(py)
}
```

**Step 2:** Register in module (add to `_native` function at line ~1988):
```rust
m.add_function(wrap_pyfunction!(rust_hac_meat, m)?)?;
m.add_function(wrap_pyfunction!(rust_dk_meat, m)?)?;
```

**Step 3:** Build and test:
```bash
uv pip install -e ".[dev]"
pytest tests/test_se.py -v
```

**Step 4:** Commit:
```bash
git add src/lib.rs polars_reg/_se.py
git commit -m "feat: add Rust HAC/DK meat functions with Bartlett kernel"
```

---

## Task 3: Add NW/DK support to `iv2sls()`

**Files:**
- Modify: `polars_reg/_iv.py`
- Test: `tests/test_iv.py`

**Step 1:** Add `time` and `bandwidth` parameters to `iv2sls()` signature (line 167):

```python
def iv2sls(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    time: str | None = None,
    bandwidth: int | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
```

**Step 2:** Add `_hac_meat` and `_dk_meat` imports at top of `_iv.py`:

```python
from polars_reg._se import (
    _clustered_meat,
    _dk_meat,
    _hac_meat,
    _interaction_codes,
    _recode_to_contiguous,
    vcov_wild_bootstrap,
)
```

**Step 3:** Update `extract_arrays` call (line 210) to pass `time`:

```python
arrays = extract_arrays(data, spec, cluster=cluster, time=time)
```

**Step 4:** Update the Rust eligibility check (line 200-206) to exclude NW/DK:

```python
    _rust_eligible = (
        _HAS_NATIVE
        and not spec.indicators
        and not any(":" in c for c in spec.exog)
        and vcov not in ("bootstrap", "wildboot", "NW", "DK")
        and (cluster or vcov in ("iid", "HC0", "HC1"))
    )
```

**Step 5:** Add NW/DK routing in the VCV section. Insert before the `elif vcov == "bootstrap"` line (before line 304):

```python
    elif vcov in ("NW", "DK"):
        if arrays.time_array is None:
            raise ValueError(f"vcov='{vcov}' requires time= parameter")
        score = X_hat * resid[:, None]
        if vcov == "NW":
            S = _hac_meat(score, arrays.time_array, bandwidth)
            dfc = n / (n - k)
        else:
            S = _dk_meat(score, arrays.time_array, bandwidth)
            T = len(np.unique(arrays.time_array))
            dfc = T / (T - 1)
        V = dfc * XhX_inv @ S @ XhX_inv
        vcov_type = vcov
        df_r = n - k - df_abs
```

**Step 6:** Write tests in `tests/test_iv.py`:

```python
def test_iv2sls_nw(iv_data_panel):
    """2SLS with Newey-West HAC standard errors."""
    result = iv2sls(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"
    assert len(result.se) > 0
    assert all(se > 0 for se in result.se)


def test_iv2sls_dk(iv_data_panel):
    """2SLS with Driscoll-Kraay standard errors."""
    result = iv2sls(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"
    assert len(result.se) > 0


def test_iv2sls_nw_requires_time(iv_data):
    """NW vcov should raise without time parameter."""
    with pytest.raises(ValueError, match="requires time"):
        iv2sls("y ~ x_exog || x_endog ~ z1 + z2", data=iv_data, vcov="NW")
```

**Step 7:** Add `iv_data_panel` fixture to `tests/conftest.py`:

```python
@pytest.fixture
def iv_data_panel() -> pl.DataFrame:
    """IV dataset with panel structure (entity + time)."""
    rng = np.random.default_rng(42)
    n_firms, n_years = 50, 20
    n = n_firms * n_years
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(n_years), n_firms)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
    x_exog = rng.standard_normal(n)
    firm_fe = rng.standard_normal(n_firms)
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + firm_fe[firm_id] + u
    return pl.DataFrame({
        "y": y, "x_endog": x_endog, "x_exog": x_exog,
        "z1": z1, "z2": z2, "firm_id": firm_id, "year_id": year_id,
    })
```

**Step 8:** Run tests:
```bash
pytest tests/test_iv.py -v
```

**Step 9:** Commit:
```bash
git add polars_reg/_iv.py tests/test_iv.py tests/conftest.py
git commit -m "feat: add NW/DK standard errors to iv2sls()"
```

---

## Task 4: Add NW/DK support to `liml()` and `gmm_iv()`

**Files:**
- Modify: `polars_reg/_gmm.py`
- Test: `tests/test_gmm.py`

**Step 1:** Add imports to `_gmm.py` (after line 17):

```python
from polars_reg._se import (
    _dk_meat,
    _hac_meat,
    vcov_clustered,
    vcov_multiway_clustered,
    vcov_pairs_bootstrap,
    vcov_robust,
    vcov_wild_bootstrap,
)
```

**Step 2:** Add `time` and `bandwidth` parameters to `liml()` (line 21):

```python
def liml(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    time: str | None = None,
    bandwidth: int | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
```

**Step 3:** Update `extract_arrays` call in `liml()` (line 44) to pass `time`:

```python
    arrays = extract_arrays(data, spec, cluster=cluster, time=time)
```

**Step 4:** Add NW/DK routing in `liml()` VCV section. Insert before `elif vcov == "iid"` (before line 151):

```python
    elif vcov in ("NW", "DK"):
        if arrays.time_array is None:
            raise ValueError(f"vcov='{vcov}' requires time= parameter")
        score = X_w * resid[:, None]
        if vcov == "NW":
            S = _hac_meat(score, arrays.time_array, bandwidth)
            dfc = n / (n - k)
        else:
            S = _dk_meat(score, arrays.time_array, bandwidth)
            T = len(np.unique(arrays.time_array))
            dfc = T / (T - 1)
        V = dfc * XwX_inv @ S @ XwX_inv
        vcov_type = vcov
        df_r = n - k
```

**Step 5:** Add `time` and `bandwidth` parameters to `gmm_iv()` (line 177):

```python
def gmm_iv(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    time: str | None = None,
    bandwidth: int | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
```

**Step 6:** Update `extract_arrays` call in `gmm_iv()` (line 198) to pass `time`:

```python
    arrays = extract_arrays(data, spec, cluster=cluster, time=time)
```

**Step 7:** Add NW/DK routing in `gmm_iv()` VCV section. For GMM, the bread is `(X'Z S^{-1} Z'X)^{-1}` and the effective score is `(X'Z S^{-1}) @ Z.T).T * resid`. Insert before the `else:` block (before line 315):

```python
    elif vcov in ("NW", "DK"):
        if arrays.time_array is None:
            raise ValueError(f"vcov='{vcov}' requires time= parameter")
        bread = np.linalg.inv(A_final)
        XZ_Sinv = XtZ @ S_final_inv
        score = (XZ_Sinv @ Z.T).T * resid[:, None]  # n x k effective scores
        if vcov == "NW":
            S_nw = _hac_meat(score, arrays.time_array, bandwidth)
            dfc = n / (n - k)
        else:
            S_nw = _dk_meat(score, arrays.time_array, bandwidth)
            T_unique = len(np.unique(arrays.time_array))
            dfc = T_unique / (T_unique - 1)
        V = dfc * bread @ S_nw @ bread / (n * n)
        vcov_type = vcov
        n_clusters_dict = None
        df_r = n - k
```

**Step 8:** Write tests in `tests/test_gmm.py`:

```python
def test_liml_nw(iv_data_panel):
    """LIML with Newey-West SEs."""
    from polars_reg._gmm import liml
    result = liml(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"
    assert all(se > 0 for se in result.se)


def test_liml_dk(iv_data_panel):
    """LIML with Driscoll-Kraay SEs."""
    from polars_reg._gmm import liml
    result = liml(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"


def test_gmm_nw(iv_data_panel):
    """GMM with Newey-West SEs."""
    from polars_reg._gmm import gmm_iv
    result = gmm_iv(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"


def test_gmm_dk(iv_data_panel):
    """GMM with Driscoll-Kraay SEs."""
    from polars_reg._gmm import gmm_iv
    result = gmm_iv(
        "y ~ x_exog || x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"
```

**Step 9:** Run tests:
```bash
pytest tests/test_gmm.py -v
```

**Step 10:** Commit:
```bash
git add polars_reg/_gmm.py tests/test_gmm.py
git commit -m "feat: add NW/DK standard errors to liml() and gmm_iv()"
```

---

## Task 5: IV with absorbed FE — test coverage

IV + FE already works (both Python and Rust paths). This task adds comprehensive tests.

**Files:**
- Modify: `tests/test_iv.py`
- Modify: `tests/conftest.py` (already has `iv_data_panel` from Task 3)

**Step 1:** Add IV+FE tests to `tests/test_iv.py`:

```python
def test_iv2sls_one_fe(iv_data_panel):
    """2SLS with one-way absorbed FE."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )
    assert result.model_type == "2SLS"
    assert result.fe_absorbed == ["firm_id"]
    assert result.df_absorbed > 0
    # Coefficient on x_endog should be close to 2.0 (DGP)
    idx = result.names.index("x_endog")
    np.testing.assert_allclose(result.coefficients[idx], 2.0, atol=0.5)


def test_iv2sls_two_fe(iv_data_panel):
    """2SLS with two-way absorbed FE."""
    result = iv2sls(
        "y ~ x_exog | firm_id + year_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )
    assert result.fe_absorbed == ["firm_id", "year_id"]
    assert result.df_absorbed > 49  # at least firm FE


def test_iv2sls_fe_cluster(iv_data_panel):
    """2SLS with FE and clustered SEs."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
        cluster=["firm_id"],
    )
    assert result.vcov_type == "cluster"
    assert "firm_id" in result.n_clusters


def test_iv2sls_fe_robust(iv_data_panel):
    """2SLS with FE and robust SEs."""
    result_iid = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )
    result_hc1 = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="HC1",
    )
    # Coefficients should be identical, SEs should differ
    np.testing.assert_allclose(result_iid.coefficients, result_hc1.coefficients, rtol=1e-10)
    assert not np.allclose(result_iid.se, result_hc1.se)


def test_iv2sls_fe_first_stage_f(iv_data_panel):
    """First-stage F should be computed with FE."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )
    assert result.first_stage_f is not None
    assert result.first_stage_f > 5


def test_iv2sls_fe_matches_manual_demean(iv_data_panel):
    """IV+FE should match manual demean-then-2SLS for coefficients."""
    # Rust/auto path
    result_fe = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
    )
    # Manual demean
    from polars_reg._demean import demean
    from polars_reg._utils import _to_codes
    codes = _to_codes(iv_data_panel["firm_id"])
    fe_dict = {"firm_id": codes}
    y = iv_data_panel["y"].to_numpy().astype(np.float64)
    x_exog = iv_data_panel["x_exog"].to_numpy().astype(np.float64)
    x_endog = iv_data_panel["x_endog"].to_numpy().astype(np.float64)
    z1 = iv_data_panel["z1"].to_numpy().astype(np.float64)
    z2 = iv_data_panel["z2"].to_numpy().astype(np.float64)
    all_vars = np.column_stack([y, x_exog, x_endog, z1, z2])
    dm = demean(all_vars, fe_dict)
    y_dm, x_exog_dm, x_endog_dm, z1_dm, z2_dm = dm[:, 0], dm[:, 1], dm[:, 2], dm[:, 3], dm[:, 4]
    # Manual 2SLS
    Z = np.column_stack([x_exog_dm, z1_dm, z2_dm])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    X_endog_hat = Z @ (ZtZ_inv @ (Z.T @ x_endog_dm.reshape(-1, 1)))
    X_hat = np.column_stack([x_exog_dm, X_endog_hat])
    X = np.column_stack([x_exog_dm, x_endog_dm])
    beta_manual = np.linalg.solve(X_hat.T @ X, X_hat.T @ y_dm)
    np.testing.assert_allclose(result_fe.coefficients, beta_manual, rtol=1e-6)


def test_iv2sls_fe_nw(iv_data_panel):
    """2SLS + FE + Newey-West SEs."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="NW",
        time="year_id",
    )
    assert result.vcov_type == "NW"
    assert result.fe_absorbed == ["firm_id"]


def test_iv2sls_fe_dk(iv_data_panel):
    """2SLS + FE + Driscoll-Kraay SEs."""
    result = iv2sls(
        "y ~ x_exog | firm_id | x_endog ~ z1 + z2",
        data=iv_data_panel,
        vcov="DK",
        time="year_id",
    )
    assert result.vcov_type == "DK"
```

**Step 2:** Run tests:
```bash
pytest tests/test_iv.py -v
```

**Step 3:** Commit:
```bash
git add tests/test_iv.py
git commit -m "test: add comprehensive IV+FE test coverage"
```

---

## Task 6: Extend `panel_re()` with full SE support

**Files:**
- Modify: `polars_reg/_panel.py`
- Test: `tests/test_panel.py`

**Step 1:** Update `panel_re()` signature (line 155) to accept all SE types:

```python
def panel_re(
    formula: str,
    data: pl.DataFrame | pl.LazyFrame,
    entity: str,
    time: str | None = None,
    vcov: str = "iid",
    cluster: list[str] | str | None = None,
    bandwidth: int | None = None,
    n_boot: int = 999,
    seed: int | None = None,
) -> RegressionResult:
    """Panel random effects (GLS) estimator.

    Args:
        formula: Formula string, e.g. "y ~ x1 + x2"
        data: Polars DataFrame or LazyFrame
        entity: Column name for entity (panel) identifier
        time: Column name for time identifier (required for NW/DK)
        vcov: "iid", "HC0", "HC1", "NW", "DK", "bootstrap", or "wildboot"
        cluster: Column name(s) for clustered SEs. Default clusters by entity.
        bandwidth: Number of lags for NW/DK. Default: Newey-West rule of thumb.
        n_boot: Bootstrap replications (default 999).
        seed: Random seed for bootstrap reproducibility.
    """
```

**Step 2:** Update `extract_arrays` call (line 180) to pass `cluster` and `time`:

```python
    if isinstance(cluster, str):
        cluster = [cluster]
    data = ensure_polars(data)
    spec = parse_formula(formula)
    arrays = extract_arrays(data, spec, cluster=cluster if cluster else None, time=time)
```

**Step 3:** Replace the VCV section (lines 241-246) with full routing. The bread is `(X_re'X_re)^{-1}`, and we use original residuals `resid = y - X @ beta` for the sandwich meat (not transformed residuals):

```python
    # Original residuals for sandwich (not quasi-demeaned)
    resid = y - X @ beta
    # Quasi-demeaned residuals for iid VCV
    resid_re = y_re - X_re @ beta

    n_clusters_dict = None
    if vcov == "bootstrap":
        V = vcov_pairs_bootstrap(X_re, y_re, n_boot=n_boot, seed=seed)
        vcov_type_str = "bootstrap"
        df_r = n - k
    elif vcov == "wildboot":
        if not cluster:
            cluster = [entity]
        cl_codes = entity_codes if cluster == [entity] else (
            data[cluster[0]].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy().astype(np.int32)
        )
        V = vcov_wild_bootstrap(X_re, resid, cl_codes, n_boot=n_boot, seed=seed)
        vcov_type_str = "wildboot"
        n_clusters_dict = {cluster[0]: len(np.unique(cl_codes))}
        df_r = n_clusters_dict[cluster[0]] - 1
    elif vcov in ("NW", "DK"):
        if arrays.time_array is None:
            raise ValueError(f"vcov='{vcov}' requires time= parameter")
        XtX_inv = np.linalg.inv(X_re.T @ X_re)
        score = X_re * resid[:, None]
        if vcov == "NW":
            S = _hac_meat(score, arrays.time_array, bandwidth)
            dfc = n / (n - k)
        else:
            S = _dk_meat(score, arrays.time_array, bandwidth)
            T_unique = len(np.unique(arrays.time_array))
            dfc = T_unique / (T_unique - 1)
        V = dfc * XtX_inv @ S @ XtX_inv
        vcov_type_str = vcov
        df_r = n - k
    elif cluster or vcov in ("HC0", "HC1"):
        if not cluster:
            cluster = [entity]
        # Need cluster codes
        cluster_arrays_list = []
        for c in cluster:
            if c == entity:
                cluster_arrays_list.append(entity_codes)
            elif arrays.cluster_arrays and c in arrays.cluster_arrays:
                cluster_arrays_list.append(arrays.cluster_arrays[c])
            else:
                cl = data[c].cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy().astype(np.int32)
                cluster_arrays_list.append(cl)

        if vcov in ("HC0", "HC1"):
            V = vcov_robust(X_re, resid, kind=vcov)
        elif len(cluster_arrays_list) == 1:
            V = vcov_clustered(X_re, resid, cluster_arrays_list[0])
        else:
            V = vcov_multiway_clustered(X_re, resid, cluster_arrays_list)
        n_clusters_dict = {c: len(np.unique(a)) for c, a in zip(cluster, cluster_arrays_list)}
        if vcov in ("HC0", "HC1"):
            vcov_type_str = vcov
            df_r = n - k
        else:
            vcov_type_str = "cluster"
            df_r = min(n_clusters_dict.values()) - 1
    else:
        V = vcov_iid(X_re, resid_re)
        vcov_type_str = "iid"
        df_r = n - k
```

**Step 4:** Add imports at top of `_panel.py`:

```python
from polars_reg._se import (
    _dk_meat,
    _hac_meat,
    vcov_clustered,
    vcov_driscoll_kraay,
    vcov_hac,
    vcov_iid,
    vcov_multiway_clustered,
    vcov_pairs_bootstrap,
    vcov_robust,
    vcov_wild_bootstrap,
)
```

**Step 5:** Update `RegressionResult` constructor to pass `n_clusters`:

```python
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
        model_type="Panel RE",
        vcov_type=vcov_type_str,
        n_clusters=n_clusters_dict,
    )
```

**Step 6:** Write tests in `tests/test_panel.py`:

```python
def test_panel_re_cluster(panel_data):
    """RE with clustered SEs by entity."""
    result = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id", cluster=["firm_id"])
    assert result.vcov_type == "cluster"
    assert "firm_id" in result.n_clusters


def test_panel_re_robust(panel_data):
    """RE with HC1 robust SEs."""
    result = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id", vcov="HC1")
    assert result.vcov_type == "HC1"
    # Robust SEs should differ from iid
    result_iid = panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id")
    assert not np.allclose(result.se, result_iid.se)


def test_panel_re_nw(panel_data):
    """RE with Newey-West SEs."""
    result = panel_re(
        "y ~ x1 + x2", data=panel_data, entity="firm_id",
        vcov="NW", time="year_id",
    )
    assert result.vcov_type == "NW"
    assert all(se > 0 for se in result.se)


def test_panel_re_dk(panel_data):
    """RE with Driscoll-Kraay SEs."""
    result = panel_re(
        "y ~ x1 + x2", data=panel_data, entity="firm_id",
        vcov="DK", time="year_id",
    )
    assert result.vcov_type == "DK"


def test_panel_re_wildboot(panel_data):
    """RE with wild cluster bootstrap SEs."""
    result = panel_re(
        "y ~ x1 + x2", data=panel_data, entity="firm_id",
        vcov="wildboot", cluster=["firm_id"], seed=42,
    )
    assert result.vcov_type == "wildboot"


def test_panel_re_nw_requires_time(panel_data):
    """NW vcov should raise without time parameter."""
    with pytest.raises(ValueError, match="requires time"):
        panel_re("y ~ x1 + x2", data=panel_data, entity="firm_id", vcov="NW")
```

**Step 7:** Run tests:
```bash
pytest tests/test_panel.py -v
```

**Step 8:** Commit:
```bash
git add polars_reg/_panel.py tests/test_panel.py
git commit -m "feat: extend panel_re() with clustered, robust, NW, DK, wildboot SEs"
```

---

## Task 7: Generate Stata parity fixtures

**Files:**
- Create: `tests/fixtures/parity_data.csv`
- Create: `tests/fixtures/stata/generate_fixtures.do`
- Create: `tests/fixtures/stata/*.csv` (~20 fixture files)

**Step 1:** Create the synthetic parity dataset. Use a Python script to generate it deterministically:

```python
# Run once: python -c "..." or add to a generate script
import numpy as np
import pandas as pd

rng = np.random.default_rng(12345)
n = 10_000
n_firms = 100
n_years = 20

firm_id = rng.integers(0, n_firms, size=n)
year_id = rng.integers(2000, 2000 + n_years, size=n)
x1 = rng.standard_normal(n)
x2 = rng.standard_normal(n)
z1 = rng.standard_normal(n)
z2 = rng.standard_normal(n)
u = rng.standard_normal(n)
x_endog = 0.5 * z1 + 0.3 * z2 + 0.8 * u
firm_fe = rng.standard_normal(n_firms)
year_fe = rng.standard_normal(n_years)
y = 2.0 + x1 - 0.5 * x2 + 1.5 * x_endog + firm_fe[firm_id] + year_fe[year_id - 2000] + u

df = pd.DataFrame({
    "y": y, "x1": x1, "x2": x2, "x_endog": x_endog,
    "z1": z1, "z2": z2, "firm_id": firm_id, "year_id": year_id,
})
df.to_csv("tests/fixtures/parity_data.csv", index=False)
```

**Step 2:** Create the Stata `.do` file at `tests/fixtures/stata/generate_fixtures.do`:

```stata
* generate_fixtures.do — Run in Stata to produce parity fixture CSVs
* Usage: stata -b do generate_fixtures.do

clear all
set more off

local base_dir = "`c(pwd)'"

* Load data
import delimited "`base_dir'/parity_data.csv", clear

* Helper program to export results
capture program drop export_results
program define export_results
    args filename
    matrix b = e(b)
    matrix V = e(V)
    local names : colnames b
    local k = colsof(b)
    local n = e(N)

    tempname fh
    file open `fh' using "`filename'", write replace
    file write `fh' "variable,coef,se,t,p" _n
    forvalues i = 1/`k' {
        local name : word `i' of `names'
        local coef = b[1, `i']
        local se = sqrt(V[`i', `i'])
        local t = `coef' / `se'
        local p = 2 * ttail(e(df_r), abs(`t'))
        file write `fh' "`name',`coef',`se',`t',`p'" _n
    }
    * Add stats row
    file write `fh' "_stat_n,`n',,,," _n
    capture local r2 = e(r2)
    if _rc == 0 {
        file write `fh' "_stat_r2,`r2',,," _n
    }
    capture local f = e(F)
    if _rc == 0 {
        file write `fh' "_stat_F,`f',,," _n
    }
    file close `fh'
end

local outdir "`base_dir'"

* ─── OLS ───
reg y x1 x2
export_results "`outdir'/ols_iid.csv"

reg y x1 x2, vce(hc2)
export_results "`outdir'/ols_hc2.csv"

reg y x1 x2, vce(hc3)
export_results "`outdir'/ols_hc3.csv"

reg y x1 x2, vce(robust)
export_results "`outdir'/ols_hc1.csv"

reg y x1 x2, vce(cluster firm_id)
export_results "`outdir'/ols_cluster.csv"

newey y x1 x2, lag(4)
export_results "`outdir'/ols_nw.csv"

* DK requires xtset
xtset firm_id year_id
xtscc y x1 x2, lag(4)
export_results "`outdir'/ols_dk.csv"

* ─── OLS + FE ───
reghdfe y x1 x2, absorb(firm_id) vce(cluster firm_id)
export_results "`outdir'/ols_fe_cluster.csv"

reghdfe y x1 x2, absorb(firm_id) vce(robust)
export_results "`outdir'/ols_fe_hc1.csv"

reghdfe y x1 x2, absorb(firm_id year_id) vce(cluster firm_id)
export_results "`outdir'/ols_2fe_cluster.csv"

* ─── 2SLS ───
ivregress 2sls y x1 (x_endog = z1 z2)
export_results "`outdir'/iv_iid.csv"

ivregress 2sls y x1 (x_endog = z1 z2), vce(robust)
export_results "`outdir'/iv_robust.csv"

ivregress 2sls y x1 (x_endog = z1 z2), vce(cluster firm_id)
export_results "`outdir'/iv_cluster.csv"

* IV + FE
ivreghdfe y x1 (x_endog = z1 z2), absorb(firm_id) cluster(firm_id)
export_results "`outdir'/iv_fe_cluster.csv"

* ─── Panel RE ───
xtset firm_id year_id
xtreg y x1 x2, re
export_results "`outdir'/re_iid.csv"

xtreg y x1 x2, re vce(cluster firm_id)
export_results "`outdir'/re_cluster.csv"

* ─── Newey (baseline HAC check) ───
newey y x1 x2, lag(8)
export_results "`outdir'/newey_lag8.csv"

di "All fixtures generated."
```

**Step 3:** Run the Stata `.do` file locally to generate fixtures, then commit. The exact command depends on your Stata installation. From the `tests/fixtures/stata/` directory:

```bash
cd tests/fixtures/stata
"/mnt/c/Program Files/Stata18/StataBE-64.exe" -b do generate_fixtures.do
cd ../../..
```

**Step 4:** Commit fixtures:
```bash
git add tests/fixtures/
git commit -m "test: add Stata parity fixture data and generation script"
```

---

## Task 8: Write Stata parity tests

**Files:**
- Modify: `tests/test_stata_parity.py`

**Step 1:** Write parametrized parity tests that load fixture CSVs and compare against polars_reg:

```python
"""Tests comparing polars_reg output against frozen Stata fixtures."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import polars_reg as pr

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "stata"
DATA_PATH = Path(__file__).parent / "fixtures" / "parity_data.csv"

# Skip all if fixtures not generated yet
pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="Parity fixtures not generated (run tests/fixtures/stata/generate_fixtures.do)",
)


@pytest.fixture(scope="module")
def parity_data():
    return pl.read_csv(str(DATA_PATH))


def load_fixture(name: str) -> pd.DataFrame:
    """Load a Stata fixture CSV. Returns DataFrame with variable, coef, se columns."""
    path = FIXTURE_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"Fixture {name}.csv not found")
    df = pd.read_csv(path)
    return df


def compare_coefs(result, fixture_df, rtol_coef=1e-6, rtol_se=1e-4):
    """Compare polars_reg result against Stata fixture."""
    # Filter out stat rows
    coef_rows = fixture_df[~fixture_df["variable"].str.startswith("_stat_")]
    for _, row in coef_rows.iterrows():
        name = row["variable"]
        if name == "_cons":
            if "_cons" in result.names:
                idx = result.names.index("_cons")
            else:
                continue
        elif name in result.names:
            idx = result.names.index(name)
        else:
            continue
        np.testing.assert_allclose(
            result.coefficients[idx], row["coef"],
            rtol=rtol_coef,
            err_msg=f"Coefficient mismatch for {name}",
        )
        np.testing.assert_allclose(
            result.se[idx], row["se"],
            rtol=rtol_se,
            err_msg=f"SE mismatch for {name}",
        )


# ─── OLS ───

def test_parity_ols_iid(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data)
    compare_coefs(result, load_fixture("ols_iid"))


def test_parity_ols_hc1(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="HC1")
    compare_coefs(result, load_fixture("ols_hc1"))


def test_parity_ols_hc2(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="HC2")
    compare_coefs(result, load_fixture("ols_hc2"))


def test_parity_ols_hc3(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="HC3")
    compare_coefs(result, load_fixture("ols_hc3"))


def test_parity_ols_cluster(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, cluster=["firm_id"])
    compare_coefs(result, load_fixture("ols_cluster"))


def test_parity_ols_nw(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="NW", time="year_id", bandwidth=4)
    compare_coefs(result, load_fixture("ols_nw"), rtol_se=1e-3)


def test_parity_ols_dk(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="DK", time="year_id", bandwidth=4)
    compare_coefs(result, load_fixture("ols_dk"), rtol_se=1e-3)


# ─── OLS + FE ───

def test_parity_ols_fe_cluster(parity_data):
    result = pr.ols("y ~ x1 + x2 | firm_id", data=parity_data, cluster=["firm_id"])
    compare_coefs(result, load_fixture("ols_fe_cluster"))


def test_parity_ols_fe_hc1(parity_data):
    result = pr.ols("y ~ x1 + x2 | firm_id", data=parity_data, vcov="HC1")
    compare_coefs(result, load_fixture("ols_fe_hc1"))


def test_parity_ols_2fe_cluster(parity_data):
    result = pr.ols("y ~ x1 + x2 | firm_id + year_id", data=parity_data, cluster=["firm_id"])
    compare_coefs(result, load_fixture("ols_2fe_cluster"))


# ─── 2SLS ───

def test_parity_iv_iid(parity_data):
    result = pr.iv2sls("y ~ x1 || x_endog ~ z1 + z2", data=parity_data)
    compare_coefs(result, load_fixture("iv_iid"))


def test_parity_iv_robust(parity_data):
    result = pr.iv2sls("y ~ x1 || x_endog ~ z1 + z2", data=parity_data, vcov="HC1")
    compare_coefs(result, load_fixture("iv_robust"))


def test_parity_iv_cluster(parity_data):
    result = pr.iv2sls("y ~ x1 || x_endog ~ z1 + z2", data=parity_data, cluster=["firm_id"])
    compare_coefs(result, load_fixture("iv_cluster"))


def test_parity_iv_fe_cluster(parity_data):
    result = pr.iv2sls(
        "y ~ x1 | firm_id | x_endog ~ z1 + z2",
        data=parity_data, cluster=["firm_id"],
    )
    compare_coefs(result, load_fixture("iv_fe_cluster"), rtol_se=1e-3)


# ─── Panel RE ───

def test_parity_re_iid(parity_data):
    result = pr.panel_re("y ~ x1 + x2", data=parity_data, entity="firm_id")
    compare_coefs(result, load_fixture("re_iid"), rtol_se=1e-3)


def test_parity_re_cluster(parity_data):
    result = pr.panel_re(
        "y ~ x1 + x2", data=parity_data, entity="firm_id", cluster=["firm_id"],
    )
    compare_coefs(result, load_fixture("re_cluster"), rtol_se=1e-3)


# ─── Newey (HAC baseline) ───

def test_parity_newey(parity_data):
    result = pr.ols("y ~ x1 + x2", data=parity_data, vcov="NW", time="year_id", bandwidth=8)
    compare_coefs(result, load_fixture("newey_lag8"), rtol_se=1e-3)
```

**Step 2:** Run tests (they'll skip if fixtures aren't generated yet):
```bash
pytest tests/test_stata_parity.py -v
```

**Step 3:** Commit:
```bash
git add tests/test_stata_parity.py
git commit -m "test: add Stata parity tests for all estimator/SE combinations"
```

---

## Task Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Factor out HAC/DK meat helpers | `_se.py`, `tests/test_se.py` |
| 2 | Rust HAC/DK meat functions | `src/lib.rs` |
| 3 | NW/DK for `iv2sls()` | `_iv.py`, `tests/test_iv.py`, `conftest.py` |
| 4 | NW/DK for `liml()`, `gmm_iv()` | `_gmm.py`, `tests/test_gmm.py` |
| 5 | IV+FE test coverage | `tests/test_iv.py` |
| 6 | `panel_re()` full SE support | `_panel.py`, `tests/test_panel.py` |
| 7 | Generate Stata fixtures | `tests/fixtures/` |
| 8 | Stata parity tests | `tests/test_stata_parity.py` |

**Recommended order:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

Tasks 1-2 are foundational (meat helpers). Tasks 3-4 use them. Task 5 is pure tests. Task 6 extends panel RE. Tasks 7-8 are the final validation batch.
