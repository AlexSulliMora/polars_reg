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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_demean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_clustered_meat, m)?)?;
    m.add_function(wrap_pyfunction!(rust_recode, m)?)?;
    Ok(())
}
