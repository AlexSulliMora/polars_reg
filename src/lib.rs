use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute group counts (equivalent to np.bincount).
fn group_counts(codes: &[i32], n_groups: usize) -> Vec<f64> {
    let mut counts = vec![0.0_f64; n_groups];
    for &c in codes {
        counts[c as usize] += 1.0;
    }
    counts
}

/// Subtract group means from a contiguous column slice, in-place.
#[inline]
fn demean_col(col: &mut [f64], codes: &[i32], denom: &[f64], n_groups: usize) {
    let mut sums = vec![0.0_f64; n_groups];
    for (i, &c) in codes.iter().enumerate() {
        sums[c as usize] += col[i];
    }
    for (i, &c) in codes.iter().enumerate() {
        let g = c as usize;
        col[i] -= sums[g] / denom[g];
    }
}

/// Column-major matrix: data stored as Vec<f64> with n rows and k columns.
/// Column j spans data[j*n .. (j+1)*n], so each column is a contiguous slice.
struct ColMajorMatrix {
    data: Vec<f64>,
    n: usize,
    k: usize,
}

impl ColMajorMatrix {
    fn from_row_major(x: &Array2<f64>) -> Self {
        let n = x.nrows();
        let k = x.ncols();
        let mut data = vec![0.0_f64; n * k];
        for j in 0..k {
            for i in 0..n {
                data[j * n + i] = x[[i, j]];
            }
        }
        ColMajorMatrix { data, n, k }
    }

    fn to_row_major(&self) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((self.n, self.k));
        for j in 0..self.k {
            for i in 0..self.n {
                x[[i, j]] = self.data[j * self.n + i];
            }
        }
        x
    }

    fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.data[j * self.n..(j + 1) * self.n]
    }

    /// Get mutable slices to all columns simultaneously (safe because they don't overlap).
    fn cols_mut(&mut self) -> Vec<&mut [f64]> {
        let n = self.n;
        let k = self.k;
        let ptr = self.data.as_mut_ptr();
        // SAFETY: each column slice is non-overlapping (j*n..(j+1)*n)
        (0..k)
            .map(|j| unsafe { std::slice::from_raw_parts_mut(ptr.add(j * n), n) })
            .collect()
    }

    /// Sum of squares of all elements.
    fn sum_sq(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum()
    }

    /// Dot product with another ColMajorMatrix.
    fn dot(&self, other: &ColMajorMatrix) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// self += alpha * other
    fn scaled_add(&mut self, alpha: f64, other: &ColMajorMatrix) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += alpha * b;
        }
    }

    /// self = other + beta * self
    fn update_from(&mut self, other: &ColMajorMatrix, beta: f64) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = *b + beta * *a;
        }
    }

    fn assign_from(&mut self, other: &ColMajorMatrix) {
        self.data.copy_from_slice(&other.data);
    }

    fn clone_cm(&self) -> ColMajorMatrix {
        ColMajorMatrix {
            data: self.data.clone(),
            n: self.n,
            k: self.k,
        }
    }

    fn sub(&self, other: &ColMajorMatrix) -> ColMajorMatrix {
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        ColMajorMatrix {
            data,
            n: self.n,
            k: self.k,
        }
    }
}

/// Subtract group means for all columns, parallelized across columns.
fn subtract_group_means_par(cm: &mut ColMajorMatrix, codes: &[i32], denom: &[f64]) {
    let n_groups = denom.len();
    let cols = cm.cols_mut();
    cols.into_par_iter().for_each(|col| {
        demean_col(col, codes, denom, n_groups);
    });
}

/// Subtract group means for all columns, sequential.
fn subtract_group_means_seq(cm: &mut ColMajorMatrix, codes: &[i32], denom: &[f64]) {
    let n_groups = denom.len();
    for j in 0..cm.k {
        demean_col(cm.col_mut(j), codes, denom, n_groups);
    }
}

/// One symmetric Kaczmarz sweep: forward then backward through FE dimensions.
fn symmetric_kaczmarz(
    cm: &mut ColMajorMatrix,
    fe_list: &[Vec<i32>],
    denoms: &[Vec<f64>],
    par: bool,
) {
    let d = fe_list.len();
    let demean_fn = if par {
        subtract_group_means_par
    } else {
        subtract_group_means_seq
    };
    // Forward
    for dim in 0..d {
        demean_fn(cm, &fe_list[dim], &denoms[dim]);
    }
    // Backward (skip last)
    for dim in (0..d - 1).rev() {
        demean_fn(cm, &fe_list[dim], &denoms[dim]);
    }
}

/// CG-accelerated demeaning with symmetric Kaczmarz.
fn demean_cg(
    x_in: &Array2<f64>,
    fe_list: &[Vec<i32>],
    n_groups_list: &[usize],
    tol: f64,
    max_iter: usize,
) -> Array2<f64> {
    let n = x_in.nrows();
    let k = x_in.ncols();
    // Parallelize when N is large enough AND k > 1
    let par = n >= 50_000 && k > 1;

    // Precompute group counts
    let denoms: Vec<Vec<f64>> = fe_list
        .iter()
        .zip(n_groups_list.iter())
        .map(|(codes, &ng)| group_counts(codes, ng))
        .collect();

    // Convert to column-major for contiguous column access
    let mut x = ColMajorMatrix::from_row_major(x_in);
    let mut tmp = x.clone_cm();

    // Initial residual: r = T(x) - x
    symmetric_kaczmarz(&mut tmp, fe_list, &denoms, par);
    let mut r = tmp.sub(&x);
    let mut u = r.clone_cm();
    let mut ssr: f64 = r.sum_sq();

    for _ in 0..max_iter {
        let x_norm: f64 = x.sum_sq();
        if ssr < tol * tol * x_norm.max(1e-16) {
            break;
        }

        // v = u - T(u)
        tmp.assign_from(&u);
        symmetric_kaczmarz(&mut tmp, fe_list, &denoms, par);
        let v = u.sub(&tmp);

        let uv: f64 = u.dot(&v);
        if uv.abs() < 1e-30 {
            break;
        }

        let alpha = ssr / uv;
        x.scaled_add(alpha, &u);
        r.scaled_add(-alpha, &v);

        let ssr_new: f64 = r.sum_sq();
        let beta = ssr_new / ssr;

        u.update_from(&r, beta);
        ssr = ssr_new;
    }

    x.to_row_major()
}

/// Subtract group means from each column of an Array2, in-place (single FE path).
fn subtract_group_means_array(x: &mut Array2<f64>, codes: &[i32], denom: &[f64]) {
    let n = x.nrows();
    let k = x.ncols();
    let n_groups = denom.len();
    for j in 0..k {
        let mut sums = vec![0.0_f64; n_groups];
        for i in 0..n {
            sums[codes[i] as usize] += x[[i, j]];
        }
        for i in 0..n {
            let g = codes[i] as usize;
            x[[i, j]] -= sums[g] / denom[g];
        }
    }
}

/// Full demean function callable from Python.
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
        // Single FE: exact in one pass, no need for CG or col-major
        let mut result = x_arr;
        let denom = group_counts(&fe_list[0], n_groups_list[0]);
        subtract_group_means_array(&mut result, &fe_list[0], &denom);
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

// ---------------------------------------------------------------------------
// Recode helpers (internal, operate on Vec<i32>)
// ---------------------------------------------------------------------------

fn recode_vec(arr: &[i32]) -> (Vec<i32>, usize) {
    if arr.is_empty() {
        return (vec![], 0);
    }
    let mn = *arr.iter().min().unwrap() as i64;
    let mx = *arr.iter().max().unwrap() as i64;
    let range = (mx - mn) as usize;
    let n = arr.len();
    if range < 2 * n {
        let mut lut = vec![-1_i32; range + 1];
        let mut next_code: i32 = 0;
        let mut codes = vec![0_i32; n];
        for i in 0..n {
            let idx = (arr[i] as i64 - mn) as usize;
            if lut[idx] < 0 {
                lut[idx] = next_code;
                next_code += 1;
            }
            codes[i] = lut[idx];
        }
        (codes, next_code as usize)
    } else {
        let mut vals: Vec<(i32, usize)> = arr.iter().copied().enumerate().map(|(i, v)| (v, i)).collect();
        vals.sort_unstable_by_key(|&(v, _)| v);
        let mut codes = vec![0_i32; n];
        let mut code: i32 = 0;
        codes[vals[0].1] = 0;
        for i in 1..n {
            if vals[i].0 != vals[i - 1].0 {
                code += 1;
            }
            codes[vals[i].1] = code;
        }
        (codes, (code + 1) as usize)
    }
}

/// Create interaction codes for multiple cluster dimensions.
fn interaction_codes(arrays: &[&[i32]]) -> (Vec<i32>, usize) {
    if arrays.len() == 1 {
        return recode_vec(arrays[0]);
    }
    let n = arrays[0].len();
    let mut combined = vec![0_i64; n];
    for i in 0..n {
        combined[i] = arrays[0][i] as i64;
    }
    for arr in &arrays[1..] {
        let arr_max = *arr.iter().max().unwrap_or(&0) as i64 + 1;
        for i in 0..n {
            combined[i] = combined[i] * arr_max + arr[i] as i64;
        }
    }
    // Recode combined to contiguous
    let i32_combined: Vec<i32> = combined.iter().map(|&v| v as i32).collect();
    recode_vec(&i32_combined)
}

/// Check if FE codes are nested within cluster codes.
fn is_nested(fe_codes: &[i32], cluster_codes: &[i32]) -> bool {
    let n_fe = *fe_codes.iter().max().unwrap_or(&0) as usize + 1;
    let mut cl_min = vec![i32::MAX; n_fe];
    let mut cl_max = vec![i32::MIN; n_fe];
    for i in 0..fe_codes.len() {
        let g = fe_codes[i] as usize;
        let c = cluster_codes[i];
        if c < cl_min[g] { cl_min[g] = c; }
        if c > cl_max[g] { cl_max[g] = c; }
    }
    cl_min.iter().zip(cl_max.iter()).all(|(mn, mx)| mn == mx)
}

/// Drop singleton groups iteratively. Returns boolean keep mask.
fn drop_singletons_mask(fe_list: &[Vec<i32>]) -> Vec<bool> {
    let n = fe_list[0].len();
    let mut keep = vec![true; n];
    let mut changed = true;
    while changed {
        changed = false;
        for codes in fe_list {
            // Count active observations per group
            let n_groups = *codes.iter().max().unwrap_or(&0) as usize + 1;
            let mut counts = vec![0_u32; n_groups];
            for i in 0..n {
                if keep[i] {
                    counts[codes[i] as usize] += 1;
                }
            }
            for i in 0..n {
                if keep[i] && counts[codes[i] as usize] == 1 {
                    keep[i] = false;
                    changed = true;
                }
            }
        }
    }
    keep
}

/// Clustered meat computation on raw slices (no ndarray overhead).
/// X is row-major n x k, resid is n, codes is n with groups 0..n_groups-1.
fn clustered_meat_raw(
    x_data: &[f64],
    n: usize,
    k: usize,
    resid: &[f64],
    codes: &[i32],
    n_groups: usize,
) -> Vec<f64> {
    // S[g, j] = sum of X[i,j] * resid[i] for i in cluster g
    let mut s_flat = vec![0.0_f64; n_groups * k]; // row-major: s_flat[g*k + j]
    for i in 0..n {
        let g = codes[i] as usize;
        let e = resid[i];
        let x_row = &x_data[i * k..(i + 1) * k];
        let s_row = &mut s_flat[g * k..(g + 1) * k];
        for j in 0..k {
            s_row[j] += x_row[j] * e;
        }
    }
    // meat = S' @ S (k x k result)
    let mut meat = vec![0.0_f64; k * k];
    for g in 0..n_groups {
        let s_row = &s_flat[g * k..(g + 1) * k];
        for j in 0..k {
            for l in j..k {
                meat[j * k + l] += s_row[j] * s_row[l];
            }
        }
    }
    // Symmetrize
    for j in 0..k {
        for l in (j + 1)..k {
            meat[l * k + j] = meat[j * k + l];
        }
    }
    meat
}

/// Solve k x k linear system Ax = b using LU decomposition (partial pivoting).
fn solve_kxk(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    // Copy A (row-major k x k) — we'll modify in place
    let mut lu = a.to_vec();
    let mut piv: Vec<usize> = (0..k).collect();

    for col in 0..k {
        // Find pivot
        let mut max_val = lu[piv[col] * k + col].abs();
        let mut max_row = col;
        for row in (col + 1)..k {
            let v = lu[piv[row] * k + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        piv.swap(col, max_row);

        let pivot = lu[piv[col] * k + col];
        if pivot.abs() < 1e-30 {
            // Singular or near-singular — return zeros
            return vec![0.0; k];
        }

        for row in (col + 1)..k {
            let factor = lu[piv[row] * k + col] / pivot;
            lu[piv[row] * k + col] = factor; // store L factor
            for j in (col + 1)..k {
                lu[piv[row] * k + j] -= factor * lu[piv[col] * k + j];
            }
        }
    }

    // Forward substitution (Ly = Pb)
    let mut y = vec![0.0_f64; k];
    for i in 0..k {
        y[i] = b[piv[i]];
        for j in 0..i {
            y[i] -= lu[piv[i] * k + j] * y[j];
        }
    }

    // Backward substitution (Ux = y)
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        x[i] = y[i];
        for j in (i + 1)..k {
            x[i] -= lu[piv[i] * k + j] * x[j];
        }
        x[i] /= lu[piv[i] * k + i];
    }
    x
}

/// Invert a k x k matrix using LU.
fn invert_kxk(a: &[f64], k: usize) -> Vec<f64> {
    let mut inv = vec![0.0_f64; k * k];
    for col in 0..k {
        let mut e = vec![0.0_f64; k];
        e[col] = 1.0;
        let x = solve_kxk(a, &e, k);
        for row in 0..k {
            inv[row * k + col] = x[row];
        }
    }
    inv
}

/// Matrix multiply: C = A @ B where A is m x p, B is p x n (all row-major flat).
fn matmul(a: &[f64], b: &[f64], m: usize, p: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for l in 0..p {
            let a_il = a[i * p + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
    c
}

/// Full OLS core: demean + solve + clustered SE, all in Rust.
///
/// Takes:
///   y (n,), X (n x k), fe_codes_list (list of i32 arrays), n_groups_list,
///   cluster_codes_list (list of i32 arrays for cluster dims, empty if no clustering),
///   tol, max_iter
///
/// Returns: (beta [k], vcov [k x k], residuals [n], r2, r2_adj,
///           n_obs, df_abs, n_singletons_dropped,
///           cluster_n_groups [per cluster dim])
#[pyfunction]
fn rust_ols_core<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    fe_codes_list: Vec<PyReadonlyArray1<'py, i32>>,
    _n_groups_list: Vec<usize>,
    cluster_codes_list: Vec<PyReadonlyArray1<'py, i32>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,   // beta
    Bound<'py, PyArray2<f64>>,   // vcov
    Bound<'py, PyArray1<f64>>,   // residuals
    f64,                          // r2
    f64,                          // r2_adj
    usize,                        // n_obs (after dropping singletons)
    usize,                        // df_abs (passed through, computed in Python)
    usize,                        // n_singletons_dropped
    Vec<usize>,                   // n_clusters per cluster dim
)> {
    let y_arr = y.as_array();
    let x_arr = x.as_array();
    let n_orig = y_arr.len();
    let k = x_arr.ncols();
    let has_fe = !fe_codes_list.is_empty();
    let has_cluster = !cluster_codes_list.is_empty();

    // Convert inputs to owned Vecs
    let mut y_vec: Vec<f64> = y_arr.iter().copied().collect();
    let mut x_flat: Vec<f64> = Vec::with_capacity(n_orig * k);
    for i in 0..n_orig {
        for j in 0..k {
            x_flat.push(x_arr[[i, j]]);
        }
    }
    let mut fe_list: Vec<Vec<i32>> = fe_codes_list
        .iter()
        .map(|a| a.as_array().to_vec())
        .collect();
    let mut cl_list: Vec<Vec<i32>> = cluster_codes_list
        .iter()
        .map(|a| a.as_array().to_vec())
        .collect();

    // --- Drop singletons ---
    let mut n_dropped = 0_usize;
    if has_fe {
        let keep = drop_singletons_mask(&fe_list);
        let n_keep: usize = keep.iter().filter(|&&b| b).count();
        if n_keep < n_orig {
            n_dropped = n_orig - n_keep;
            // Filter all arrays
            let mut new_y = Vec::with_capacity(n_keep);
            let mut new_x = Vec::with_capacity(n_keep * k);
            for i in 0..n_orig {
                if keep[i] {
                    new_y.push(y_vec[i]);
                    for j in 0..k {
                        new_x.push(x_flat[i * k + j]);
                    }
                }
            }
            y_vec = new_y;
            x_flat = new_x;
            for fe in &mut fe_list {
                let mut new_fe = Vec::with_capacity(n_keep);
                for i in 0..n_orig {
                    if keep[i] {
                        new_fe.push(fe[i]);
                    }
                }
                *fe = new_fe;
            }
            for cl in &mut cl_list {
                let mut new_cl = Vec::with_capacity(n_keep);
                for i in 0..n_orig {
                    if keep[i] {
                        new_cl.push(cl[i]);
                    }
                }
                *cl = new_cl;
            }
        }
    }

    let n = y_vec.len();

    // --- Demean ---
    if has_fe {
        // Build Array2 with [y, X] columns for demeaning
        let total_cols = 1 + k;
        let mut all_arr = Array2::<f64>::zeros((n, total_cols));
        for i in 0..n {
            all_arr[[i, 0]] = y_vec[i];
            for j in 0..k {
                all_arr[[i, j + 1]] = x_flat[i * k + j];
            }
        }

        let ng_list: Vec<usize> = fe_list.iter().map(|fe| *fe.iter().max().unwrap_or(&0) as usize + 1).collect();

        let demeaned = if fe_list.len() == 1 {
            let denom = group_counts(&fe_list[0], ng_list[0]);
            subtract_group_means_array(&mut all_arr, &fe_list[0], &denom);
            all_arr
        } else {
            demean_cg(&all_arr, &fe_list, &ng_list, tol, max_iter)
        };

        // Extract back
        for i in 0..n {
            y_vec[i] = demeaned[[i, 0]];
            for j in 0..k {
                x_flat[i * k + j] = demeaned[[i, j + 1]];
            }
        }
    }

    // --- OLS solve: beta = (X'X)^{-1} X'y ---
    // X'X (k x k)
    let mut xtx = vec![0.0_f64; k * k];
    for i in 0..n {
        let row = &x_flat[i * k..(i + 1) * k];
        for j in 0..k {
            for l in j..k {
                xtx[j * k + l] += row[j] * row[l];
            }
        }
    }
    for j in 0..k {
        for l in (j + 1)..k {
            xtx[l * k + j] = xtx[j * k + l];
        }
    }

    // X'y (k)
    let mut xty = vec![0.0_f64; k];
    for i in 0..n {
        let row = &x_flat[i * k..(i + 1) * k];
        let yi = y_vec[i];
        for j in 0..k {
            xty[j] += row[j] * yi;
        }
    }

    let beta = solve_kxk(&xtx, &xty, k);

    // --- Residuals ---
    let mut resid = vec![0.0_f64; n];
    for i in 0..n {
        let mut pred = 0.0;
        for j in 0..k {
            pred += x_flat[i * k + j] * beta[j];
        }
        resid[i] = y_vec[i] - pred;
    }

    // --- R-squared ---
    let ss_res: f64 = resid.iter().map(|r| r * r).sum();
    let y_mean: f64 = y_vec.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y_vec.iter().map(|yi| (yi - y_mean) * (yi - y_mean)).sum();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    // Compute absorbed DoF using union-find (replaces Python scipy path)
    let df_abs = if has_fe { absorbed_dof_internal(&fe_list) } else { 0 };

    // --- Variance-covariance ---
    let xtx_inv = invert_kxk(&xtx, k);
    let vcov: Vec<f64>;
    let mut cluster_n_groups: Vec<usize> = Vec::new();

    if has_cluster {
        // Compute nesting: for each FE dim, check if nested in any cluster dim
        let mut non_nested_dof = 0_usize;
        if has_fe {
            for fe in &fe_list {
                let mut nested = false;
                for cl in &cl_list {
                    if is_nested(fe, cl) {
                        nested = true;
                        break;
                    }
                }
                if !nested {
                    let n_groups = *fe.iter().max().unwrap_or(&0) as usize + 1;
                    non_nested_dof += n_groups - 1;
                }
            }
        }

        // Recode each cluster dimension
        let cl_recoded: Vec<(Vec<i32>, usize)> = cl_list.iter().map(|cl| recode_vec(cl)).collect();
        for (_, g) in &cl_recoded {
            cluster_n_groups.push(*g);
        }

        // CGM inclusion-exclusion
        let d = cl_recoded.len();
        let mut v_total = vec![0.0_f64; k * k];

        // Use global dfc if FE are present (reghdfe-style)
        let g_min = *cluster_n_groups.iter().min().unwrap();

        for size in 1..=d {
            let sign = if size % 2 == 1 { 1.0 } else { -1.0 };
            // Iterate over all subsets of given size
            let subsets = combinations(d, size);
            for subset in subsets {
                let (codes, g) = if subset.len() == 1 {
                    cl_recoded[subset[0]].clone()
                } else {
                    let arrays: Vec<&[i32]> = subset.iter().map(|&idx| cl_recoded[idx].0.as_slice()).collect();
                    interaction_codes(&arrays)
                };

                let meat = clustered_meat_raw(&x_flat, n, k, &resid, &codes, g);

                let dfc = if has_fe {
                    // reghdfe-style: single dfc using g_min for all CGM terms
                    // matches Python df_a_non_nested >= 0 branch
                    (g_min as f64 / (g_min as f64 - 1.0)) * (n as f64 / (n as f64 - non_nested_dof as f64 - k as f64))
                } else {
                    // Standard: per-term G/(G-1) * (N-1)/(N-k)
                    (g as f64 / (g as f64 - 1.0)) * ((n as f64 - 1.0) / (n as f64 - k as f64))
                };

                // V += sign * dfc * xtx_inv @ meat @ xtx_inv
                let tmp = matmul(&xtx_inv, &meat, k, k, k);
                let term = matmul(&tmp, &xtx_inv, k, k, k);
                for idx in 0..(k * k) {
                    v_total[idx] += sign * dfc * term[idx];
                }
            }
        }
        vcov = v_total;
    } else {
        // iid: sigma^2 * (X'X)^{-1}
        let sigma2 = ss_res / (n - k) as f64;
        vcov = xtx_inv.iter().map(|v| v * sigma2).collect();
    }

    // Convert outputs to numpy
    let beta_arr = Array1::from_vec(beta);
    let mut vcov_arr = Array2::<f64>::zeros((k, k));
    for j in 0..k {
        for l in 0..k {
            vcov_arr[[j, l]] = vcov[j * k + l];
        }
    }
    let resid_arr = Array1::from_vec(resid);

    Ok((
        beta_arr.into_pyarray(py),
        vcov_arr.into_pyarray(py),
        resid_arr.into_pyarray(py),
        r2,
        {
            let denom = n as f64 - k as f64 - df_abs as f64;
            if denom > 0.0 { 1.0 - (1.0 - r2) * (n as f64 - 1.0) / denom } else { 0.0 }
        },
        n,
        df_abs,
        n_dropped,
        cluster_n_groups,
    ))
}

/// Generate all combinations of `size` elements from 0..n.
fn combinations(n: usize, size: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut combo = vec![0_usize; size];
    fn recurse(n: usize, size: usize, start: usize, depth: usize, combo: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if depth == size {
            result.push(combo.clone());
            return;
        }
        for i in start..n {
            combo[depth] = i;
            recurse(n, size, i + 1, depth + 1, combo, result);
        }
    }
    recurse(n, size, 0, 0, &mut combo, &mut result);
    result
}

/// Union-Find (disjoint set) with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry { return; }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }

    fn count_components(&mut self) -> usize {
        let n = self.parent.len();
        let mut seen = vec![false; n];
        let mut count = 0;
        for i in 0..n {
            let r = self.find(i);
            if !seen[r] {
                seen[r] = true;
                count += 1;
            }
        }
        count
    }
}

/// Count connected components in bipartite graph of two FE dimensions using union-find.
fn connected_components_uf(codes_a: &[i32], n_a: usize, codes_b: &[i32], n_b: usize) -> usize {
    let total = n_a + n_b;
    let mut uf = UnionFind::new(total);
    for i in 0..codes_a.len() {
        let a = codes_a[i] as usize;
        let b = n_a + codes_b[i] as usize;
        uf.union(a, b);
    }
    uf.count_components()
}

/// Compute degrees of freedom absorbed by fixed effects.
/// Single FE: number of groups.
/// Two+ FE: sum of groups minus connected components (pairwise method).
fn absorbed_dof_internal(fe_list: &[Vec<i32>]) -> usize {
    let n_groups: Vec<usize> = fe_list.iter()
        .map(|fe| *fe.iter().max().unwrap_or(&0) as usize + 1)
        .collect();
    let mut total_dof = n_groups[0];

    for i in 1..fe_list.len() {
        total_dof += n_groups[i];
        let mut max_components = 0;
        for j in 0..i {
            let c = connected_components_uf(&fe_list[j], n_groups[j], &fe_list[i], n_groups[i]);
            max_components = max_components.max(c);
        }
        total_dof -= max_components;
    }

    total_dof
}

/// Python-callable absorbed DoF computation using union-find.
#[pyfunction]
fn rust_absorbed_dof<'py>(
    _py: Python<'py>,
    fe_codes_list: Vec<PyReadonlyArray1<'py, i32>>,
) -> PyResult<usize> {
    let fe_list: Vec<Vec<i32>> = fe_codes_list.iter()
        .map(|a| a.as_slice().unwrap().to_vec())
        .collect();
    Ok(absorbed_dof_internal(&fe_list))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_demean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_clustered_meat, m)?)?;
    m.add_function(wrap_pyfunction!(rust_recode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ols_core, m)?)?;
    m.add_function(wrap_pyfunction!(rust_absorbed_dof, m)?)?;
    Ok(())
}
