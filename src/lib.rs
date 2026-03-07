use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Demeaning primitives
// ---------------------------------------------------------------------------

/// Compute group counts (equivalent to np.bincount).
fn group_counts(codes: &[i32], n_groups: usize) -> Vec<f64> {
    let mut counts = vec![0.0_f64; n_groups];
    for &c in codes {
        counts[c as usize] += 1.0;
    }
    counts
}

/// Subtract group means from a contiguous column slice, in-place.
/// Uses chunked parallel reduction for the scatter-add when n is large.
#[inline]
fn demean_col(col: &mut [f64], codes: &[i32], denom: &[f64], n_groups: usize) {
    let n = col.len();
    // Parallel reduction threshold: use parallel when N >= 200K
    // and n_groups is large enough to avoid false sharing
    let use_par = n >= 200_000 && n_groups >= 32;

    let means = if use_par {
        // Chunk the data and reduce in parallel
        let n_chunks = rayon::current_num_threads().max(1);
        let chunk_size = (n + n_chunks - 1) / n_chunks;

        let chunk_sums: Vec<Vec<f64>> = (0..n_chunks)
            .into_par_iter()
            .map(|ci| {
                let start = ci * chunk_size;
                let end = (start + chunk_size).min(n);
                let mut local = vec![0.0_f64; n_groups];
                for i in start..end {
                    local[codes[i] as usize] += col[i];
                }
                local
            })
            .collect();

        // Merge
        let mut sums = vec![0.0_f64; n_groups];
        for cs in &chunk_sums {
            for (s, c) in sums.iter_mut().zip(cs.iter()) {
                *s += *c;
            }
        }
        // Compute means
        for g in 0..n_groups {
            sums[g] /= denom[g];
        }
        sums
    } else {
        let mut sums = vec![0.0_f64; n_groups];
        for i in 0..n {
            sums[codes[i] as usize] += col[i];
        }
        for g in 0..n_groups {
            sums[g] /= denom[g];
        }
        sums
    };

    // Subtract means
    for i in 0..n {
        col[i] -= means[codes[i] as usize];
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
    /// Create from row-major flat slice (n rows, k cols).
    fn from_row_major_flat(x: &[f64], n: usize, k: usize) -> Self {
        let mut data = vec![0.0_f64; n * k];
        for j in 0..k {
            for i in 0..n {
                data[j * n + i] = x[i * k + j];
            }
        }
        ColMajorMatrix { data, n, k }
    }

    fn from_row_major_array(x: &Array2<f64>) -> Self {
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

    /// Write back to row-major flat slice.
    fn to_row_major_flat(&self, out: &mut [f64]) {
        let n = self.n;
        let k = self.k;
        for j in 0..k {
            for i in 0..n {
                out[i * k + j] = self.data[j * n + i];
            }
        }
    }

    fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.data[j * self.n..(j + 1) * self.n]
    }

    /// Get mutable slices to all columns simultaneously (safe because they don't overlap).
    fn cols_mut(&mut self) -> Vec<&mut [f64]> {
        let n = self.n;
        let k = self.k;
        let ptr = self.data.as_mut_ptr();
        (0..k)
            .map(|j| unsafe { std::slice::from_raw_parts_mut(ptr.add(j * n), n) })
            .collect()
    }

    fn sum_sq(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum()
    }

    fn dot(&self, other: &ColMajorMatrix) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    fn scaled_add(&mut self, alpha: f64, other: &ColMajorMatrix) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += alpha * b;
        }
    }

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

/// Subtract group means for all columns, using per-column parallelism inside demean_col.
fn subtract_group_means_cm(cm: &mut ColMajorMatrix, codes: &[i32], denom: &[f64]) {
    let n_groups = denom.len();
    for j in 0..cm.k {
        demean_col(cm.col_mut(j), codes, denom, n_groups);
    }
}

/// One symmetric Kaczmarz sweep: forward then backward through FE dimensions.
fn symmetric_kaczmarz(cm: &mut ColMajorMatrix, fe_list: &[&[i32]], denoms: &[Vec<f64>]) {
    let d = fe_list.len();
    for dim in 0..d {
        subtract_group_means_cm(cm, fe_list[dim], &denoms[dim]);
    }
    for dim in (0..d - 1).rev() {
        subtract_group_means_cm(cm, fe_list[dim], &denoms[dim]);
    }
}

/// CG-accelerated demeaning with symmetric Kaczmarz. Operates on borrowed FE slices.
fn demean_cg_slices(
    x_in: &ColMajorMatrix,
    fe_list: &[&[i32]],
    n_groups_list: &[usize],
    tol: f64,
    max_iter: usize,
) -> ColMajorMatrix {
    let denoms: Vec<Vec<f64>> = fe_list
        .iter()
        .zip(n_groups_list.iter())
        .map(|(codes, &ng)| group_counts(codes, ng))
        .collect();

    let mut x = x_in.clone_cm();
    let mut tmp = x.clone_cm();

    symmetric_kaczmarz(&mut tmp, fe_list, &denoms);
    let mut r = tmp.sub(&x);
    let mut u = r.clone_cm();
    let mut ssr: f64 = r.sum_sq();

    for _ in 0..max_iter {
        let x_norm: f64 = x.sum_sq();
        if ssr < tol * tol * x_norm.max(1e-16) {
            break;
        }

        tmp.assign_from(&u);
        symmetric_kaczmarz(&mut tmp, fe_list, &denoms);
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

    x
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
    let n_fe = fe_codes_list.len();

    if n_fe == 1 {
        let mut result = x_arr;
        let codes = fe_codes_list[0].as_slice().unwrap();
        let denom = group_counts(codes, n_groups_list[0]);
        subtract_group_means_array(&mut result, codes, &denom);
        result.into_pyarray(py)
    } else {
        // Borrow slices directly from numpy — zero copy for FE codes
        let fe_slices: Vec<&[i32]> = fe_codes_list
            .iter()
            .map(|a| a.as_slice().unwrap())
            .collect();
        let cm_in = ColMajorMatrix::from_row_major_array(&x_arr);
        let result = demean_cg_slices(&cm_in, &fe_slices, &n_groups_list, tol, max_iter);
        result.to_row_major().into_pyarray(py)
    }
}

// ---------------------------------------------------------------------------
// Clustered SE helpers
// ---------------------------------------------------------------------------

/// Clustered meat computation: sum_g (s_g s_g') where s_g = sum of X[i]*e[i] for i in group g.
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

    let mut s_mat = Array2::<f64>::zeros((n_groups, k));
    for i in 0..n {
        let g = codes_arr[i] as usize;
        let e = resid_arr[i];
        for j in 0..k {
            s_mat[[g, j]] += x_arr[[i, j]] * e;
        }
    }

    let meat = s_mat.t().dot(&s_mat);
    meat.into_pyarray(py)
}

/// Recode arbitrary integer array to contiguous 0..G-1 codes.
#[pyfunction]
fn rust_recode<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
) -> (Bound<'py, PyArray1<i32>>, usize) {
    let a = arr.as_array();
    let n = a.len();

    let mut mn = i64::MAX;
    let mut mx = i64::MIN;
    for &v in a.iter() {
        if v < mn { mn = v; }
        if v > mx { mx = v; }
    }

    let range = (mx - mn) as usize;
    if range < 2 * n {
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
        (codes.into_pyarray(py), next_code as usize)
    } else {
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
        (codes.into_pyarray(py), (code + 1) as usize)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers (operate on slices, no ndarray/numpy overhead)
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
    let i32_combined: Vec<i32> = combined.iter().map(|&v| v as i32).collect();
    recode_vec(&i32_combined)
}

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

fn drop_singletons_mask(fe_list: &[&[i32]]) -> Vec<bool> {
    let n = fe_list[0].len();
    let mut keep = vec![true; n];
    let mut changed = true;
    while changed {
        changed = false;
        for codes in fe_list {
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

/// Clustered meat computation on raw slices.
fn clustered_meat_raw(
    x_data: &[f64],
    n: usize,
    k: usize,
    resid: &[f64],
    codes: &[i32],
    n_groups: usize,
) -> Vec<f64> {
    let mut s_flat = vec![0.0_f64; n_groups * k];
    for i in 0..n {
        let g = codes[i] as usize;
        let e = resid[i];
        let x_row = &x_data[i * k..(i + 1) * k];
        let s_row = &mut s_flat[g * k..(g + 1) * k];
        for j in 0..k {
            s_row[j] += x_row[j] * e;
        }
    }
    let mut meat = vec![0.0_f64; k * k];
    for g in 0..n_groups {
        let s_row = &s_flat[g * k..(g + 1) * k];
        for j in 0..k {
            for l in j..k {
                meat[j * k + l] += s_row[j] * s_row[l];
            }
        }
    }
    for j in 0..k {
        for l in (j + 1)..k {
            meat[l * k + j] = meat[j * k + l];
        }
    }
    meat
}

fn solve_kxk(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut lu = a.to_vec();
    let mut piv: Vec<usize> = (0..k).collect();

    for col in 0..k {
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
            return vec![0.0; k];
        }

        for row in (col + 1)..k {
            let factor = lu[piv[row] * k + col] / pivot;
            lu[piv[row] * k + col] = factor;
            for j in (col + 1)..k {
                lu[piv[row] * k + j] -= factor * lu[piv[col] * k + j];
            }
        }
    }

    let mut y = vec![0.0_f64; k];
    for i in 0..k {
        y[i] = b[piv[i]];
        for j in 0..i {
            y[i] -= lu[piv[i] * k + j] * y[j];
        }
    }

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

// ---------------------------------------------------------------------------
// Union-Find for absorbed DoF
// ---------------------------------------------------------------------------

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

fn connected_components_uf(codes_a: &[i32], n_a: usize, codes_b: &[i32], n_b: usize) -> usize {
    let total = n_a + n_b;
    let mut uf = UnionFind::new(total);
    for i in 0..codes_a.len() {
        uf.union(codes_a[i] as usize, n_a + codes_b[i] as usize);
    }
    uf.count_components()
}

fn absorbed_dof_internal(fe_list: &[&[i32]]) -> usize {
    let n_groups: Vec<usize> = fe_list.iter()
        .map(|fe| *fe.iter().max().unwrap_or(&0) as usize + 1)
        .collect();
    let mut total_dof = n_groups[0];

    for i in 1..fe_list.len() {
        total_dof += n_groups[i];
        let mut max_components = 0;
        for j in 0..i {
            let c = connected_components_uf(fe_list[j], n_groups[j], fe_list[i], n_groups[i]);
            max_components = max_components.max(c);
        }
        total_dof -= max_components;
    }
    total_dof
}

#[pyfunction]
fn rust_absorbed_dof<'py>(
    _py: Python<'py>,
    fe_codes_list: Vec<PyReadonlyArray1<'py, i32>>,
) -> PyResult<usize> {
    let fe_slices: Vec<&[i32]> = fe_codes_list.iter()
        .map(|a| a.as_slice().unwrap())
        .collect();
    Ok(absorbed_dof_internal(&fe_slices))
}

// ---------------------------------------------------------------------------
// rust_ols_core — full OLS pipeline with zero-copy for read-only arrays
// ---------------------------------------------------------------------------

/// Full OLS core: demean + solve + clustered SE, all in Rust.
/// FE and cluster code arrays are borrowed directly from numpy (zero-copy).
/// Only y and X are copied since they get mutated during demeaning.
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
    usize,                        // n_obs
    usize,                        // df_abs
    usize,                        // n_singletons_dropped
    Vec<usize>,                   // n_clusters per cluster dim
)> {
    let x_arr = x.as_array();
    let n_orig = x_arr.nrows();
    let k = x_arr.ncols();
    let has_fe = !fe_codes_list.is_empty();
    let has_cluster = !cluster_codes_list.is_empty();

    // Borrow FE and cluster codes directly from numpy (zero-copy reads)
    let fe_slices: Vec<&[i32]> = fe_codes_list.iter()
        .map(|a| a.as_slice().unwrap())
        .collect();
    let cl_slices: Vec<&[i32]> = cluster_codes_list.iter()
        .map(|a| a.as_slice().unwrap())
        .collect();

    // Copy y (will be mutated by demeaning)
    let y_slice = y.as_slice().unwrap();
    let mut y_vec: Vec<f64> = y_slice.to_vec();

    // Copy X row-major flat (will be mutated by demeaning)
    let mut x_flat: Vec<f64> = Vec::with_capacity(n_orig * k);
    // Use standard layout if available for faster copy
    if let Some(x_slice) = x_arr.as_slice() {
        x_flat.extend_from_slice(x_slice);
    } else {
        for i in 0..n_orig {
            for j in 0..k {
                x_flat.push(x_arr[[i, j]]);
            }
        }
    }

    // --- Drop singletons ---
    let mut n_dropped = 0_usize;
    // After singleton drop, we need owned FE/cluster Vecs only if rows were removed
    let mut fe_owned: Vec<Vec<i32>> = Vec::new();
    let mut cl_owned: Vec<Vec<i32>> = Vec::new();
    let mut dropped = false;

    if has_fe {
        let keep = drop_singletons_mask(&fe_slices);
        let n_keep: usize = keep.iter().filter(|&&b| b).count();
        if n_keep < n_orig {
            n_dropped = n_orig - n_keep;
            dropped = true;

            // Filter y and X
            let mut new_y = Vec::with_capacity(n_keep);
            let mut new_x = Vec::with_capacity(n_keep * k);
            for i in 0..n_orig {
                if keep[i] {
                    new_y.push(y_vec[i]);
                    new_x.extend_from_slice(&x_flat[i * k..(i + 1) * k]);
                }
            }
            y_vec = new_y;
            x_flat = new_x;

            // Filter FE codes (must own since we're subsetting)
            for codes in &fe_slices {
                let mut new_codes = Vec::with_capacity(n_keep);
                for i in 0..n_orig {
                    if keep[i] {
                        new_codes.push(codes[i]);
                    }
                }
                fe_owned.push(new_codes);
            }
            // Filter cluster codes
            for codes in &cl_slices {
                let mut new_codes = Vec::with_capacity(n_keep);
                for i in 0..n_orig {
                    if keep[i] {
                        new_codes.push(codes[i]);
                    }
                }
                cl_owned.push(new_codes);
            }
        }
    }

    // Build working FE/cluster slice references (either borrowed or owned)
    let fe_work: Vec<&[i32]> = if dropped {
        fe_owned.iter().map(|v| v.as_slice()).collect()
    } else {
        fe_slices
    };
    let cl_work: Vec<&[i32]> = if dropped {
        cl_owned.iter().map(|v| v.as_slice()).collect()
    } else {
        cl_slices
    };

    let n = y_vec.len();

    // --- Demean ---
    if has_fe {
        let ng_list: Vec<usize> = fe_work.iter()
            .map(|fe| *fe.iter().max().unwrap_or(&0) as usize + 1)
            .collect();

        if fe_work.len() == 1 {
            // Single FE: exact in one pass using ColMajorMatrix for consistency
            let denom = group_counts(fe_work[0], ng_list[0]);
            // Demean y
            demean_col(&mut y_vec, fe_work[0], &denom, ng_list[0]);
            // Demean X columns
            let mut cm = ColMajorMatrix::from_row_major_flat(&x_flat, n, k);
            for j in 0..k {
                demean_col(cm.col_mut(j), fe_work[0], &denom, ng_list[0]);
            }
            cm.to_row_major_flat(&mut x_flat);
        } else {
            // Multi-FE CG: stack [y, X] into column-major, demean, extract back
            let total_cols = 1 + k;
            let mut cm_data = vec![0.0_f64; n * total_cols];
            // Col 0 = y
            cm_data[..n].copy_from_slice(&y_vec);
            // Cols 1..k+1 = X (transpose from row-major)
            for j in 0..k {
                for i in 0..n {
                    cm_data[(j + 1) * n + i] = x_flat[i * k + j];
                }
            }
            let cm_in = ColMajorMatrix { data: cm_data, n, k: total_cols };
            let cm_out = demean_cg_slices(&cm_in, &fe_work, &ng_list, tol, max_iter);

            // Extract back
            y_vec.copy_from_slice(&cm_out.data[..n]);
            for j in 0..k {
                for i in 0..n {
                    x_flat[i * k + j] = cm_out.data[(j + 1) * n + i];
                }
            }
        }
    }

    // --- OLS solve: beta = (X'X)^{-1} X'y ---
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

    let df_abs = if has_fe { absorbed_dof_internal(&fe_work) } else { 0 };

    // --- Variance-covariance ---
    let xtx_inv = invert_kxk(&xtx, k);
    let vcov: Vec<f64>;
    let mut cluster_n_groups: Vec<usize> = Vec::new();

    if has_cluster {
        let mut non_nested_dof = 0_usize;
        if has_fe {
            for fe in &fe_work {
                let mut nested = false;
                for cl in &cl_work {
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

        let cl_recoded: Vec<(Vec<i32>, usize)> = cl_work.iter().map(|cl| recode_vec(cl)).collect();
        for (_, g) in &cl_recoded {
            cluster_n_groups.push(*g);
        }

        let d = cl_recoded.len();
        let mut v_total = vec![0.0_f64; k * k];
        let g_min = *cluster_n_groups.iter().min().unwrap();

        for size in 1..=d {
            let sign = if size % 2 == 1 { 1.0 } else { -1.0 };
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
                    (g_min as f64 / (g_min as f64 - 1.0)) * (n as f64 / (n as f64 - non_nested_dof as f64 - k as f64))
                } else {
                    (g as f64 / (g as f64 - 1.0)) * ((n as f64 - 1.0) / (n as f64 - k as f64))
                };

                let tmp = matmul(&xtx_inv, &meat, k, k, k);
                let term = matmul(&tmp, &xtx_inv, k, k, k);
                for idx in 0..(k * k) {
                    v_total[idx] += sign * dfc * term[idx];
                }
            }
        }
        vcov = v_total;
    } else {
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

// ---------------------------------------------------------------------------
// rust_ols_from_arrays — accepts individual column arrays, skips Python extraction overhead
// ---------------------------------------------------------------------------

/// OLS with FE absorption — accepts individual column arrays directly.
/// Avoids Python-side column_stack, astype, and ascontiguousarray overhead.
///
/// y_col: f64 array (n,) — dependent variable
/// x_cols: list of f64 arrays (n,) — regressor columns
/// x_names: list of str — regressor names
/// fe_cols: list of i32 arrays (n,) — FE code columns
/// fe_names: list of str — FE names
/// cl_cols: list of i32 arrays (n,) — cluster code columns (empty if no clustering)
/// cl_names: list of str — cluster names
#[pyfunction]
fn rust_ols_from_arrays<'py>(
    py: Python<'py>,
    y_col: PyReadonlyArray1<'py, f64>,
    x_cols: Vec<PyReadonlyArray1<'py, f64>>,
    x_names: Vec<String>,
    fe_cols: Vec<PyReadonlyArray1<'py, i32>>,
    fe_names: Vec<String>,
    cl_cols: Vec<PyReadonlyArray1<'py, i32>>,
    cl_names: Vec<String>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,   // beta
    Bound<'py, PyArray2<f64>>,   // vcov
    Bound<'py, PyArray1<f64>>,   // residuals
    f64,                          // r2
    f64,                          // r2_adj
    usize,                        // n_obs
    usize,                        // df_abs
    usize,                        // n_singletons_dropped
    Vec<usize>,                   // n_clusters per cluster dim
    Vec<String>,                  // x_names (after any dropped columns)
    Vec<String>,                  // fe_names
    Vec<String>,                  // cl_names
)> {
    let y_src = y_col.as_slice().unwrap();
    let n_orig = y_src.len();
    let k = x_cols.len();
    let has_fe = !fe_cols.is_empty();
    let has_cluster = !cl_cols.is_empty();

    // Borrow all input slices — zero copy
    let x_slices: Vec<&[f64]> = x_cols.iter().map(|a| a.as_slice().unwrap()).collect();
    let fe_slices: Vec<&[i32]> = fe_cols.iter().map(|a| a.as_slice().unwrap()).collect();
    let cl_slices: Vec<&[i32]> = cl_cols.iter().map(|a| a.as_slice().unwrap()).collect();

    // Copy y (mutated by demeaning)
    let mut y_vec: Vec<f64> = y_src.to_vec();

    // Build X row-major flat from individual columns (replaces Python column_stack)
    let mut x_flat: Vec<f64> = Vec::with_capacity(n_orig * k);
    for i in 0..n_orig {
        for j in 0..k {
            x_flat.push(x_slices[j][i]);
        }
    }

    // --- Drop singletons ---
    let mut n_dropped = 0_usize;
    let mut fe_owned: Vec<Vec<i32>> = Vec::new();
    let mut cl_owned: Vec<Vec<i32>> = Vec::new();
    let mut dropped = false;

    if has_fe {
        let keep = drop_singletons_mask(&fe_slices);
        let n_keep: usize = keep.iter().filter(|&&b| b).count();
        if n_keep < n_orig {
            n_dropped = n_orig - n_keep;
            dropped = true;

            let mut new_y = Vec::with_capacity(n_keep);
            let mut new_x = Vec::with_capacity(n_keep * k);
            for i in 0..n_orig {
                if keep[i] {
                    new_y.push(y_vec[i]);
                    new_x.extend_from_slice(&x_flat[i * k..(i + 1) * k]);
                }
            }
            y_vec = new_y;
            x_flat = new_x;

            for codes in &fe_slices {
                let new_codes: Vec<i32> = (0..n_orig).filter(|&i| keep[i]).map(|i| codes[i]).collect();
                fe_owned.push(new_codes);
            }
            for codes in &cl_slices {
                let new_codes: Vec<i32> = (0..n_orig).filter(|&i| keep[i]).map(|i| codes[i]).collect();
                cl_owned.push(new_codes);
            }
        }
    }

    let fe_work: Vec<&[i32]> = if dropped {
        fe_owned.iter().map(|v| v.as_slice()).collect()
    } else {
        fe_slices
    };
    let cl_work: Vec<&[i32]> = if dropped {
        cl_owned.iter().map(|v| v.as_slice()).collect()
    } else {
        cl_slices
    };

    let n = y_vec.len();

    // --- Demean ---
    if has_fe {
        let ng_list: Vec<usize> = fe_work.iter()
            .map(|fe| *fe.iter().max().unwrap_or(&0) as usize + 1)
            .collect();

        if fe_work.len() == 1 {
            let denom = group_counts(fe_work[0], ng_list[0]);
            demean_col(&mut y_vec, fe_work[0], &denom, ng_list[0]);
            let mut cm = ColMajorMatrix::from_row_major_flat(&x_flat, n, k);
            for j in 0..k {
                demean_col(cm.col_mut(j), fe_work[0], &denom, ng_list[0]);
            }
            cm.to_row_major_flat(&mut x_flat);
        } else {
            let total_cols = 1 + k;
            let mut cm_data = vec![0.0_f64; n * total_cols];
            cm_data[..n].copy_from_slice(&y_vec);
            for j in 0..k {
                for i in 0..n {
                    cm_data[(j + 1) * n + i] = x_flat[i * k + j];
                }
            }
            let cm_in = ColMajorMatrix { data: cm_data, n, k: total_cols };
            let cm_out = demean_cg_slices(&cm_in, &fe_work, &ng_list, tol, max_iter);
            y_vec.copy_from_slice(&cm_out.data[..n]);
            for j in 0..k {
                for i in 0..n {
                    x_flat[i * k + j] = cm_out.data[(j + 1) * n + i];
                }
            }
        }
    }

    // --- OLS solve ---
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

    let mut xty = vec![0.0_f64; k];
    for i in 0..n {
        let row = &x_flat[i * k..(i + 1) * k];
        let yi = y_vec[i];
        for j in 0..k {
            xty[j] += row[j] * yi;
        }
    }

    let beta = solve_kxk(&xtx, &xty, k);

    let mut resid = vec![0.0_f64; n];
    for i in 0..n {
        let mut pred = 0.0;
        for j in 0..k {
            pred += x_flat[i * k + j] * beta[j];
        }
        resid[i] = y_vec[i] - pred;
    }

    let ss_res: f64 = resid.iter().map(|r| r * r).sum();
    let y_mean: f64 = y_vec.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y_vec.iter().map(|yi| (yi - y_mean) * (yi - y_mean)).sum();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    let df_abs = if has_fe { absorbed_dof_internal(&fe_work) } else { 0 };

    // --- Variance-covariance ---
    let xtx_inv = invert_kxk(&xtx, k);
    let vcov: Vec<f64>;
    let mut cluster_n_groups: Vec<usize> = Vec::new();

    if has_cluster {
        let mut non_nested_dof = 0_usize;
        if has_fe {
            for fe in &fe_work {
                let mut nested = false;
                for cl in &cl_work {
                    if is_nested(fe, cl) {
                        nested = true;
                        break;
                    }
                }
                if !nested {
                    let ng = *fe.iter().max().unwrap_or(&0) as usize + 1;
                    non_nested_dof += ng - 1;
                }
            }
        }

        let cl_recoded: Vec<(Vec<i32>, usize)> = cl_work.iter().map(|cl| recode_vec(cl)).collect();
        for (_, g) in &cl_recoded {
            cluster_n_groups.push(*g);
        }

        let d = cl_recoded.len();
        let mut v_total = vec![0.0_f64; k * k];
        let g_min = *cluster_n_groups.iter().min().unwrap();

        for size in 1..=d {
            let sign = if size % 2 == 1 { 1.0 } else { -1.0 };
            for subset in combinations(d, size) {
                let (codes, g) = if subset.len() == 1 {
                    cl_recoded[subset[0]].clone()
                } else {
                    let arrays: Vec<&[i32]> = subset.iter().map(|&idx| cl_recoded[idx].0.as_slice()).collect();
                    interaction_codes(&arrays)
                };

                let meat = clustered_meat_raw(&x_flat, n, k, &resid, &codes, g);

                let dfc = if has_fe {
                    (g_min as f64 / (g_min as f64 - 1.0)) * (n as f64 / (n as f64 - non_nested_dof as f64 - k as f64))
                } else {
                    (g as f64 / (g as f64 - 1.0)) * ((n as f64 - 1.0) / (n as f64 - k as f64))
                };

                let tmp = matmul(&xtx_inv, &meat, k, k, k);
                let term = matmul(&tmp, &xtx_inv, k, k, k);
                for idx in 0..(k * k) {
                    v_total[idx] += sign * dfc * term[idx];
                }
            }
        }
        vcov = v_total;
    } else {
        let sigma2 = ss_res / (n - k) as f64;
        vcov = xtx_inv.iter().map(|v| v * sigma2).collect();
    }

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
        x_names,
        fe_names,
        cl_names,
    ))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_demean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_clustered_meat, m)?)?;
    m.add_function(wrap_pyfunction!(rust_recode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ols_core, m)?)?;
    m.add_function(wrap_pyfunction!(rust_absorbed_dof, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ols_from_arrays, m)?)?;
    Ok(())
}
