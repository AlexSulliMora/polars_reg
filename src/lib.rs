use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute group counts (equivalent to np.bincount).
fn group_counts(codes: &[i32], n_groups: usize) -> Vec<f64> {
    let mut counts = vec![0.0_f64; n_groups];
    for &c in codes {
        counts[c as usize] += 1.0;
    }
    counts
}

/// Subtract group means from each column of x, in-place.
/// x is n x k, codes is n, denom is G.
fn subtract_group_means(x: &mut Array2<f64>, codes: &[i32], denom: &[f64]) {
    let n = x.nrows();
    let k = x.ncols();
    let n_groups = denom.len();

    for j in 0..k {
        // Sum by group
        let mut sums = vec![0.0_f64; n_groups];
        for i in 0..n {
            sums[codes[i] as usize] += x[[i, j]];
        }
        // Compute means and subtract
        for i in 0..n {
            let g = codes[i] as usize;
            x[[i, j]] -= sums[g] / denom[g];
        }
    }
}

/// One symmetric Kaczmarz sweep: forward then backward through FE dimensions.
fn symmetric_kaczmarz(
    x: &mut Array2<f64>,
    fe_list: &[Vec<i32>],
    denoms: &[Vec<f64>],
) {
    let d = fe_list.len();
    // Forward
    for dim in 0..d {
        subtract_group_means(x, &fe_list[dim], &denoms[dim]);
    }
    // Backward (skip last)
    for dim in (0..d - 1).rev() {
        subtract_group_means(x, &fe_list[dim], &denoms[dim]);
    }
}

/// CG-accelerated demeaning with symmetric Kaczmarz.
/// Returns demeaned array.
fn demean_cg(
    x_in: &Array2<f64>,
    fe_list: &[Vec<i32>],
    n_groups_list: &[usize],
    tol: f64,
    max_iter: usize,
) -> Array2<f64> {
    // Precompute group counts
    let denoms: Vec<Vec<f64>> = fe_list
        .iter()
        .zip(n_groups_list.iter())
        .map(|(codes, &ng)| group_counts(codes, ng))
        .collect();

    let mut x = x_in.clone();
    let mut tmp = x.clone();

    // Initial residual: r = T(x) - x
    symmetric_kaczmarz(&mut tmp, fe_list, &denoms);
    let mut r = &tmp - &x;
    let mut u = r.clone();
    let mut ssr: f64 = r.iter().map(|v| v * v).sum();

    for _ in 0..max_iter {
        let x_norm: f64 = x.iter().map(|v| v * v).sum();
        if ssr < tol * tol * x_norm.max(1e-16) {
            break;
        }

        // v = u - T(u)
        tmp.assign(&u);
        symmetric_kaczmarz(&mut tmp, fe_list, &denoms);
        let v = &u - &tmp;

        let uv: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        if uv.abs() < 1e-30 {
            break;
        }

        let alpha = ssr / uv;
        x.scaled_add(alpha, &u);
        r.scaled_add(-alpha, &v);

        let ssr_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = ssr_new / ssr;

        // u = r + beta * u
        u.mapv_inplace(|ui| ui * beta);
        u += &r;
        ssr = ssr_new;
    }

    x
}

/// Full demean function callable from Python.
/// X: n x k array
/// fe_codes_list: list of 1D int32 arrays (FE group codes)
/// n_groups_list: list of group counts per FE dimension
/// tol: convergence tolerance
/// max_iter: max iterations
#[pyfunction]
fn rust_demean<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    fe_codes_list: Vec<PyReadonlyArray1<'py, i32>>,
    n_groups_list: Vec<usize>,
    tol: f64,
    max_iter: usize,
) -> Bound<'py, PyArray2<f64>> {
    let x_arr = x.as_array().to_owned();

    let fe_list: Vec<Vec<i32>> = fe_codes_list
        .iter()
        .map(|a| a.as_array().to_vec())
        .collect();

    let n_fe = fe_list.len();

    if n_fe == 1 {
        // Single FE: exact in one pass
        let mut result = x_arr;
        let denom = group_counts(&fe_list[0], n_groups_list[0]);
        subtract_group_means(&mut result, &fe_list[0], &denom);
        result.into_pyarray(py)
    } else {
        let result = demean_cg(&x_arr, &fe_list, &n_groups_list, tol, max_iter);
        result.into_pyarray(py)
    }
}

/// Clustered meat computation: sum_g (s_g s_g') where s_g = sum of X[i]*e[i] for i in group g.
/// Returns k x k matrix.
#[pyfunction]
fn rust_clustered_meat<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    resid: PyReadonlyArray1<'py, f64>,
    codes: PyReadonlyArray1<'py, i32>,
    n_groups: usize,
) -> Bound<'py, PyArray2<f64>> {
    let x_arr = x.as_array();
    let resid_arr = resid.as_array();
    let codes_arr = codes.as_array();
    let n = x_arr.nrows();
    let k = x_arr.ncols();

    // Aggregate scores by cluster: S[g, j] = sum of X[i,j] * resid[i] for i in cluster g
    let mut s_mat = Array2::<f64>::zeros((n_groups, k));
    for i in 0..n {
        let g = codes_arr[i] as usize;
        let e = resid_arr[i];
        for j in 0..k {
            s_mat[[g, j]] += x_arr[[i, j]] * e;
        }
    }

    // meat = S' @ S
    let meat = s_mat.t().dot(&s_mat);
    meat.into_pyarray(py)
}

/// Recode arbitrary integer array to contiguous 0..G-1 codes.
/// Returns (codes, n_groups).
#[pyfunction]
fn rust_recode<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
) -> (Bound<'py, PyArray1<i32>>, usize) {
    let a = arr.as_array();
    let n = a.len();

    // Find min/max
    let mut mn = i64::MAX;
    let mut mx = i64::MIN;
    for &v in a.iter() {
        if v < mn { mn = v; }
        if v > mx { mx = v; }
    }

    let range = (mx - mn) as usize;
    if range < 2 * n {
        // Dense: use lookup table
        let mut lut = vec![-1_i32; range + 1];
        let mut next_code: i32 = 0;
        let mut codes = Array1::<i32>::zeros(n);
        for i in 0..n {
            let idx = (a[i] - mn) as usize;
            if lut[idx] < 0 {
                lut[idx] = next_code;
                next_code += 1;
            }
            codes[i] = lut[idx];
        }
        let n_groups = next_code as usize;
        (codes.into_pyarray(py), n_groups)
    } else {
        // Sparse: sort-based approach
        let mut vals: Vec<(i64, usize)> = a.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
        vals.sort_unstable_by_key(|&(v, _)| v);

        let mut codes = Array1::<i32>::zeros(n);
        let mut code: i32 = 0;
        codes[vals[0].1] = 0;
        for i in 1..n {
            if vals[i].0 != vals[i - 1].0 {
                code += 1;
            }
            codes[vals[i].1] = code;
        }
        let n_groups = (code + 1) as usize;
        (codes.into_pyarray(py), n_groups)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_demean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_clustered_meat, m)?)?;
    m.add_function(wrap_pyfunction!(rust_recode, m)?)?;
    Ok(())
}
