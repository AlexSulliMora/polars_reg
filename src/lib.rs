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
        if !ssr_new.is_finite() {
            break;
        }
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

/// Re-index codes to be contiguous (0, 1, 2, ...) after filtering.
/// Prevents phantom zero-count groups that cause numerical instability.
fn reindex_codes(codes: &mut [i32]) {
    if codes.is_empty() {
        return;
    }
    let max_code = *codes.iter().max().unwrap() as usize;
    let mut mapping = vec![-1i32; max_code + 1];
    let mut next_id = 0i32;
    for &c in codes.iter() {
        if mapping[c as usize] < 0 {
            mapping[c as usize] = next_id;
            next_id += 1;
        }
    }
    for c in codes.iter_mut() {
        *c = mapping[*c as usize];
    }
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
            return vec![f64::NAN; k];
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
    k_adj: bool,
    g_adj: bool,
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
            // Re-index FE codes to be contiguous after singleton removal
            for codes in &mut fe_owned {
                reindex_codes(codes);
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

                let g_adj_factor = if g_adj {
                    if has_fe {
                        g_min as f64 / (g_min as f64 - 1.0)
                    } else {
                        g as f64 / (g as f64 - 1.0)
                    }
                } else {
                    1.0
                };
                let k_adj_factor = if k_adj {
                    if has_fe {
                        n as f64 / (n as f64 - non_nested_dof as f64 - k as f64)
                    } else {
                        (n as f64 - 1.0) / (n as f64 - k as f64)
                    }
                } else {
                    1.0
                };
                let dfc = g_adj_factor * k_adj_factor;

                let tmp = matmul(&xtx_inv, &meat, k, k, k);
                let term = matmul(&tmp, &xtx_inv, k, k, k);
                for idx in 0..(k * k) {
                    v_total[idx] += sign * dfc * term[idx];
                }
            }
        }
        vcov = v_total;
    } else {
        // iid VCV: sigma² = e'e/(n-k-df_abs) when k_adj=true, e'e/n when k_adj=false
        let sigma2 = if k_adj {
            let denom = n as f64 - k as f64 - df_abs as f64;
            if denom > 0.0 { ss_res / denom } else { 0.0 }
        } else {
            ss_res / n as f64
        };
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
    vcov_type: String,
    k_adj: bool,
    g_adj: bool,
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
            // Re-index FE codes to be contiguous after singleton removal
            for codes in &mut fe_owned {
                reindex_codes(codes);
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

                let g_adj_factor = if g_adj {
                    if has_fe {
                        g_min as f64 / (g_min as f64 - 1.0)
                    } else {
                        g as f64 / (g as f64 - 1.0)
                    }
                } else {
                    1.0
                };
                let k_adj_factor = if k_adj {
                    if has_fe {
                        n as f64 / (n as f64 - non_nested_dof as f64 - k as f64)
                    } else {
                        (n as f64 - 1.0) / (n as f64 - k as f64)
                    }
                } else {
                    1.0
                };
                let dfc = g_adj_factor * k_adj_factor;

                let tmp = matmul(&xtx_inv, &meat, k, k, k);
                let term = matmul(&tmp, &xtx_inv, k, k, k);
                for idx in 0..(k * k) {
                    v_total[idx] += sign * dfc * term[idx];
                }
            }
        }
        vcov = v_total;
    } else if vcov_type == "iid" {
        let sigma2 = if k_adj {
            ss_res / (n as f64 - k as f64 - df_abs as f64)
        } else {
            ss_res / n as f64
        };
        vcov = xtx_inv.iter().map(|v| v * sigma2).collect();
    } else {
        // HC0, HC1, HC2, HC3
        vcov = sandwich_vcov(&x_flat, &resid, &xtx_inv, n, k, &vcov_type, df_abs, k_adj);
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

// ---------------------------------------------------------------------------
// rust_iv2sls — full 2SLS pipeline: demean + two-stage solve + SE
// ---------------------------------------------------------------------------

/// 2SLS IV estimation entirely in Rust.
/// Accepts individual column arrays for y, exog, endog, instruments, FE, clusters.
#[pyfunction]
fn rust_iv2sls<'py>(
    py: Python<'py>,
    y_col: PyReadonlyArray1<'py, f64>,
    x_exog_cols: Vec<PyReadonlyArray1<'py, f64>>,
    x_endog_cols: Vec<PyReadonlyArray1<'py, f64>>,
    z_excl_cols: Vec<PyReadonlyArray1<'py, f64>>,
    x_names: Vec<String>,         // exog names
    endog_names: Vec<String>,     // endog names
    fe_cols: Vec<PyReadonlyArray1<'py, i32>>,
    fe_names: Vec<String>,
    cl_cols: Vec<PyReadonlyArray1<'py, i32>>,
    cl_names: Vec<String>,
    tol: f64,
    max_iter: usize,
    vcov_type: String,
    add_intercept: bool,
    k_adj: bool,
    g_adj: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,   // beta
    Bound<'py, PyArray2<f64>>,   // vcov
    Bound<'py, PyArray1<f64>>,   // residuals
    f64,                          // r2
    f64,                          // r2_adj
    f64,                          // first_stage_f
    usize,                        // n_obs
    usize,                        // df_abs
    usize,                        // n_dropped
    Vec<usize>,                   // n_clusters per cluster dim
    Vec<String>,                  // final names (exog + endog)
)> {
    let y_src = y_col.as_slice().unwrap();
    let n_orig = y_src.len();
    let has_fe = !fe_cols.is_empty();
    let has_cluster = !cl_cols.is_empty();

    // Borrow input slices
    let exog_slices: Vec<&[f64]> = x_exog_cols.iter().map(|a| a.as_slice().unwrap()).collect();
    let endog_slices: Vec<&[f64]> = x_endog_cols.iter().map(|a| a.as_slice().unwrap()).collect();
    let z_excl_slices: Vec<&[f64]> = z_excl_cols.iter().map(|a| a.as_slice().unwrap()).collect();
    let fe_slices: Vec<&[i32]> = fe_cols.iter().map(|a| a.as_slice().unwrap()).collect();
    let cl_slices: Vec<&[i32]> = cl_cols.iter().map(|a| a.as_slice().unwrap()).collect();

    let k_exog_base = exog_slices.len();
    let k_endog = endog_slices.len();
    let k_z_excl = z_excl_slices.len();
    let k_exog = if add_intercept { k_exog_base + 1 } else { k_exog_base };
    let k = k_exog + k_endog;
    let k_z = k_exog + k_z_excl;  // total instruments = exog + excluded

    // Copy y
    let mut y_vec: Vec<f64> = y_src.to_vec();

    // Build exog row-major (n_orig, k_exog) — includes intercept if requested
    let mut exog_flat: Vec<f64> = Vec::with_capacity(n_orig * k_exog);
    for i in 0..n_orig {
        for j in 0..k_exog_base {
            exog_flat.push(exog_slices[j][i]);
        }
        if add_intercept {
            exog_flat.push(1.0);
        }
    }

    // Build endog row-major (n_orig, k_endog)
    let mut endog_flat: Vec<f64> = Vec::with_capacity(n_orig * k_endog);
    for i in 0..n_orig {
        for j in 0..k_endog {
            endog_flat.push(endog_slices[j][i]);
        }
    }

    // Build excluded instruments row-major (n_orig, k_z_excl)
    let mut z_excl_flat: Vec<f64> = Vec::with_capacity(n_orig * k_z_excl);
    for i in 0..n_orig {
        for j in 0..k_z_excl {
            z_excl_flat.push(z_excl_slices[j][i]);
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
            let mut new_exog = Vec::with_capacity(n_keep * k_exog);
            let mut new_endog = Vec::with_capacity(n_keep * k_endog);
            let mut new_z = Vec::with_capacity(n_keep * k_z_excl);
            for i in 0..n_orig {
                if keep[i] {
                    new_y.push(y_vec[i]);
                    new_exog.extend_from_slice(&exog_flat[i * k_exog..(i + 1) * k_exog]);
                    new_endog.extend_from_slice(&endog_flat[i * k_endog..(i + 1) * k_endog]);
                    new_z.extend_from_slice(&z_excl_flat[i * k_z_excl..(i + 1) * k_z_excl]);
                }
            }
            y_vec = new_y;
            exog_flat = new_exog;
            endog_flat = new_endog;
            z_excl_flat = new_z;

            for codes in &fe_slices {
                let new_codes: Vec<i32> = (0..n_orig).filter(|&i| keep[i]).map(|i| codes[i]).collect();
                fe_owned.push(new_codes);
            }
            // Re-index FE codes to be contiguous after singleton removal
            for codes in &mut fe_owned {
                reindex_codes(codes);
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

    // --- Demean all arrays if FE present ---
    if has_fe {
        let ng_list: Vec<usize> = fe_work.iter()
            .map(|fe| *fe.iter().max().unwrap_or(&0) as usize + 1)
            .collect();

        if fe_work.len() == 1 {
            let denom = group_counts(fe_work[0], ng_list[0]);
            // Demean y
            demean_col(&mut y_vec, fe_work[0], &denom, ng_list[0]);
            // Demean exog
            let mut cm = ColMajorMatrix::from_row_major_flat(&exog_flat, n, k_exog);
            for j in 0..k_exog {
                demean_col(cm.col_mut(j), fe_work[0], &denom, ng_list[0]);
            }
            cm.to_row_major_flat(&mut exog_flat);
            // Demean endog
            let mut cm_end = ColMajorMatrix::from_row_major_flat(&endog_flat, n, k_endog);
            for j in 0..k_endog {
                demean_col(cm_end.col_mut(j), fe_work[0], &denom, ng_list[0]);
            }
            cm_end.to_row_major_flat(&mut endog_flat);
            // Demean excluded instruments
            let mut cm_z = ColMajorMatrix::from_row_major_flat(&z_excl_flat, n, k_z_excl);
            for j in 0..k_z_excl {
                demean_col(cm_z.col_mut(j), fe_work[0], &denom, ng_list[0]);
            }
            cm_z.to_row_major_flat(&mut z_excl_flat);
        } else {
            // Stack all into one matrix for CG demeaning
            let total_cols = 1 + k_exog + k_endog + k_z_excl;
            let mut cm_data = vec![0.0_f64; n * total_cols];
            // Col 0 = y
            cm_data[..n].copy_from_slice(&y_vec);
            // Cols 1..k_exog+1 = exog
            for j in 0..k_exog {
                for i in 0..n {
                    cm_data[(1 + j) * n + i] = exog_flat[i * k_exog + j];
                }
            }
            // Cols k_exog+1..k_exog+1+k_endog = endog
            for j in 0..k_endog {
                for i in 0..n {
                    cm_data[(1 + k_exog + j) * n + i] = endog_flat[i * k_endog + j];
                }
            }
            // Remaining cols = excluded instruments
            for j in 0..k_z_excl {
                for i in 0..n {
                    cm_data[(1 + k_exog + k_endog + j) * n + i] = z_excl_flat[i * k_z_excl + j];
                }
            }
            let cm_in = ColMajorMatrix { data: cm_data, n, k: total_cols };
            let cm_out = demean_cg_slices(&cm_in, &fe_work, &ng_list, tol, max_iter);

            // Extract back
            y_vec.copy_from_slice(&cm_out.data[..n]);
            for j in 0..k_exog {
                for i in 0..n {
                    exog_flat[i * k_exog + j] = cm_out.data[(1 + j) * n + i];
                }
            }
            for j in 0..k_endog {
                for i in 0..n {
                    endog_flat[i * k_endog + j] = cm_out.data[(1 + k_exog + j) * n + i];
                }
            }
            for j in 0..k_z_excl {
                for i in 0..n {
                    z_excl_flat[i * k_z_excl + j] = cm_out.data[(1 + k_exog + k_endog + j) * n + i];
                }
            }
        }
    }

    // --- Build Z = [exog, z_excl] row-major (n, k_z) ---
    let mut z_flat: Vec<f64> = Vec::with_capacity(n * k_z);
    for i in 0..n {
        z_flat.extend_from_slice(&exog_flat[i * k_exog..(i + 1) * k_exog]);
        z_flat.extend_from_slice(&z_excl_flat[i * k_z_excl..(i + 1) * k_z_excl]);
    }

    // --- Stage 1: X_endog_hat = Z (Z'Z)^{-1} Z' X_endog ---
    // Compute Z'Z (k_z x k_z)
    let mut ztz = vec![0.0_f64; k_z * k_z];
    for i in 0..n {
        let zrow = &z_flat[i * k_z..(i + 1) * k_z];
        for j in 0..k_z {
            for l in j..k_z {
                ztz[j * k_z + l] += zrow[j] * zrow[l];
            }
        }
    }
    for j in 0..k_z { for l in (j + 1)..k_z { ztz[l * k_z + j] = ztz[j * k_z + l]; } }

    let ztz_inv = invert_kxk(&ztz, k_z);

    // Compute Z' X_endog (k_z x k_endog)
    let mut zt_xend = vec![0.0_f64; k_z * k_endog];
    for i in 0..n {
        let zrow = &z_flat[i * k_z..(i + 1) * k_z];
        let erow = &endog_flat[i * k_endog..(i + 1) * k_endog];
        for j in 0..k_z {
            for l in 0..k_endog {
                zt_xend[j * k_endog + l] += zrow[j] * erow[l];
            }
        }
    }

    // (Z'Z)^{-1} Z' X_endog = pi (k_z x k_endog)
    let pi = matmul(&ztz_inv, &zt_xend, k_z, k_z, k_endog);

    // X_endog_hat = Z * pi (n x k_endog)
    let mut endog_hat_flat = vec![0.0_f64; n * k_endog];
    for i in 0..n {
        let zrow = &z_flat[i * k_z..(i + 1) * k_z];
        for l in 0..k_endog {
            let mut val = 0.0;
            for j in 0..k_z {
                val += zrow[j] * pi[j * k_endog + l];
            }
            endog_hat_flat[i * k_endog + l] = val;
        }
    }

    // --- First-stage F-stat (partial F-test, first endog variable) ---
    // Restricted: x_end ~ exog
    let mut exog_t_exog = vec![0.0_f64; k_exog * k_exog];
    for i in 0..n {
        let row = &exog_flat[i * k_exog..(i + 1) * k_exog];
        for j in 0..k_exog { for l in j..k_exog { exog_t_exog[j * k_exog + l] += row[j] * row[l]; } }
    }
    for j in 0..k_exog { for l in (j + 1)..k_exog { exog_t_exog[l * k_exog + j] = exog_t_exog[j * k_exog + l]; } }
    let mut exog_t_xend0 = vec![0.0_f64; k_exog];
    for i in 0..n {
        let row = &exog_flat[i * k_exog..(i + 1) * k_exog];
        let e0 = endog_flat[i * k_endog];
        for j in 0..k_exog { exog_t_xend0[j] += row[j] * e0; }
    }
    let beta_r = solve_kxk(&exog_t_exog, &exog_t_xend0, k_exog);
    let mut ss_r = 0.0_f64;
    for i in 0..n {
        let row = &exog_flat[i * k_exog..(i + 1) * k_exog];
        let mut pred = 0.0;
        for j in 0..k_exog { pred += row[j] * beta_r[j]; }
        let r = endog_flat[i * k_endog] - pred;
        ss_r += r * r;
    }
    // Unrestricted: x_end ~ Z
    let mut zt_xend0 = vec![0.0_f64; k_z];
    for i in 0..n {
        let zrow = &z_flat[i * k_z..(i + 1) * k_z];
        let e0 = endog_flat[i * k_endog];
        for j in 0..k_z { zt_xend0[j] += zrow[j] * e0; }
    }
    let beta_u = solve_kxk(&ztz, &zt_xend0, k_z);
    let mut ss_u = 0.0_f64;
    for i in 0..n {
        let zrow = &z_flat[i * k_z..(i + 1) * k_z];
        let mut pred = 0.0;
        for j in 0..k_z { pred += zrow[j] * beta_u[j]; }
        let r = endog_flat[i * k_endog] - pred;
        ss_u += r * r;
    }
    let q = k_z_excl as f64;
    let first_stage_f = if ss_u > 0.0 && q > 0.0 {
        ((ss_r - ss_u) / q) / (ss_u / (n as f64 - k_exog as f64 - q))
    } else {
        0.0
    };

    // --- Stage 2: X = [exog, endog], X_hat = [exog, endog_hat] ---
    // beta = (X_hat'X)^{-1} X_hat'y
    let mut x_flat = Vec::with_capacity(n * k);
    let mut xhat_flat = Vec::with_capacity(n * k);
    for i in 0..n {
        x_flat.extend_from_slice(&exog_flat[i * k_exog..(i + 1) * k_exog]);
        x_flat.extend_from_slice(&endog_flat[i * k_endog..(i + 1) * k_endog]);
        xhat_flat.extend_from_slice(&exog_flat[i * k_exog..(i + 1) * k_exog]);
        xhat_flat.extend_from_slice(&endog_hat_flat[i * k_endog..(i + 1) * k_endog]);
    }

    // X_hat'X (k x k)
    let mut xhx = vec![0.0_f64; k * k];
    for i in 0..n {
        let xh_row = &xhat_flat[i * k..(i + 1) * k];
        let x_row = &x_flat[i * k..(i + 1) * k];
        for j in 0..k {
            for l in 0..k {
                xhx[j * k + l] += xh_row[j] * x_row[l];
            }
        }
    }
    // X_hat'y
    let mut xhy = vec![0.0_f64; k];
    for i in 0..n {
        let xh_row = &xhat_flat[i * k..(i + 1) * k];
        let yi = y_vec[i];
        for j in 0..k { xhy[j] += xh_row[j] * yi; }
    }

    let beta = solve_kxk(&xhx, &xhy, k);
    if beta.iter().any(|v| v.is_nan()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Singular matrix in 2SLS"));
    }

    // Residuals: e = y - X*beta (using actual X, not X_hat)
    let mut resid = vec![0.0_f64; n];
    for i in 0..n {
        let x_row = &x_flat[i * k..(i + 1) * k];
        let mut pred = 0.0;
        for j in 0..k { pred += x_row[j] * beta[j]; }
        resid[i] = y_vec[i] - pred;
    }

    // R-squared
    let ss_res: f64 = resid.iter().map(|r| r * r).sum();
    let y_mean: f64 = y_vec.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y_vec.iter().map(|yi| (yi - y_mean) * (yi - y_mean)).sum();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    let df_abs = if has_fe { absorbed_dof_internal(&fe_work) } else { 0 };
    let r2_adj = {
        let denom = n as f64 - k as f64 - df_abs as f64;
        if denom > 0.0 { 1.0 - (1.0 - r2) * (n as f64 - 1.0) / denom } else { 0.0 }
    };

    // --- Variance-covariance: bread = (X_hat'X)^{-1}, meat uses X_hat ---
    let xhx_inv = invert_kxk(&xhx, k);
    let vcov: Vec<f64>;
    let mut cluster_n_groups: Vec<usize> = Vec::new();

    if has_cluster {
        let cl_recoded: Vec<(Vec<i32>, usize)> = cl_work.iter().map(|cl| recode_vec(cl)).collect();
        for (_, g) in &cl_recoded {
            cluster_n_groups.push(*g);
        }
        let d = cl_recoded.len();
        let mut v_total = vec![0.0_f64; k * k];

        for size in 1..=d {
            let sign = if size % 2 == 1 { 1.0 } else { -1.0 };
            for subset in combinations(d, size) {
                let (codes, g) = if subset.len() == 1 {
                    cl_recoded[subset[0]].clone()
                } else {
                    let arrays: Vec<&[i32]> = subset.iter().map(|&idx| cl_recoded[idx].0.as_slice()).collect();
                    interaction_codes(&arrays)
                };
                // Use X_hat for clustered meat in IV
                let meat = clustered_meat_raw(&xhat_flat, n, k, &resid, &codes, g);
                let g_adj_factor = if g_adj { g as f64 / (g as f64 - 1.0) } else { 1.0 };
                let k_adj_factor = if k_adj { (n as f64 - 1.0) / (n as f64 - k as f64) } else { 1.0 };
                let dfc = g_adj_factor * k_adj_factor;
                let tmp = matmul(&xhx_inv, &meat, k, k, k);
                let term = matmul(&tmp, &xhx_inv, k, k, k);
                for idx in 0..(k * k) {
                    v_total[idx] += sign * dfc * term[idx];
                }
            }
        }
        vcov = v_total;
    } else if vcov_type == "iid" {
        // IV iid: sigma² = e'e/(n-k) when k_adj=true, e'e/n when k_adj=false
        let sigma2 = if k_adj {
            ss_res / (n as f64 - k as f64 - df_abs as f64)
        } else {
            ss_res / n as f64
        };
        vcov = xhx_inv.iter().map(|v| v * sigma2).collect();
    } else {
        // HC0 or HC1 — k_adj controls HC1 scaling
        vcov = sandwich_vcov(&xhat_flat, &resid, &xhx_inv, n, k, &vcov_type, df_abs, k_adj);
    }

    // Build final names
    let mut final_names = x_names;
    if add_intercept {
        final_names.push("_cons".to_string());
    }
    final_names.extend(endog_names);

    let beta_arr = Array1::from_vec(beta);
    let mut vcov_arr = Array2::<f64>::zeros((k, k));
    for j in 0..k { for l in 0..k { vcov_arr[[j, l]] = vcov[j * k + l]; } }
    let resid_arr = Array1::from_vec(resid);

    Ok((
        beta_arr.into_pyarray(py),
        vcov_arr.into_pyarray(py),
        resid_arr.into_pyarray(py),
        r2,
        r2_adj,
        first_stage_f,
        n,
        df_abs,
        n_dropped,
        cluster_n_groups,
        final_names,
    ))
}

// ---------------------------------------------------------------------------
// rust_ols_nofe — full OLS pipeline for no-FE case, all SE types
// ---------------------------------------------------------------------------

/// Compute sandwich VCV: (X'X)^{-1} meat (X'X)^{-1} where meat depends on vcov_type.
/// x_flat is row-major (n, k). Returns flat k*k VCV.
/// df_abs: additional absorbed degrees of freedom (e.g., from fixed effects).
/// k_adj: if true, apply n/(n-k-df_abs) scaling for HC1; if false, HC1 == HC0.
fn sandwich_vcov(
    x_flat: &[f64],
    resid: &[f64],
    xtx_inv: &[f64],
    n: usize,
    k: usize,
    vcov_type: &str,
    df_abs: usize,
    k_adj: bool,
) -> Vec<f64> {
    match vcov_type {
        "HC0" | "HC1" => {
            // meat = X' diag(e²) X
            let mut meat = vec![0.0_f64; k * k];
            for i in 0..n {
                let row = &x_flat[i * k..(i + 1) * k];
                let e2 = resid[i] * resid[i];
                for j in 0..k {
                    for l in j..k {
                        meat[j * k + l] += row[j] * row[l] * e2;
                    }
                }
            }
            for j in 0..k {
                for l in (j + 1)..k {
                    meat[l * k + j] = meat[j * k + l];
                }
            }
            let tmp = matmul(xtx_inv, &meat, k, k, k);
            let mut v = matmul(&tmp, xtx_inv, k, k, k);
            if vcov_type == "HC1" && k_adj {
                let scale = n as f64 / (n as f64 - k as f64 - df_abs as f64);
                for val in v.iter_mut() {
                    *val *= scale;
                }
            }
            v
        }
        "HC2" | "HC3" => {
            // hat_ii = x_i' (X'X)^{-1} x_i
            let mut hat = vec![0.0_f64; n];
            for i in 0..n {
                let row = &x_flat[i * k..(i + 1) * k];
                let mut h = 0.0;
                for j in 0..k {
                    for l in 0..k {
                        h += row[j] * xtx_inv[j * k + l] * row[l];
                    }
                }
                hat[i] = h;
            }
            let mut meat = vec![0.0_f64; k * k];
            for i in 0..n {
                let row = &x_flat[i * k..(i + 1) * k];
                let e2 = resid[i] * resid[i];
                let w = if vcov_type == "HC2" {
                    e2 / (1.0 - hat[i])
                } else {
                    e2 / ((1.0 - hat[i]) * (1.0 - hat[i]))
                };
                for j in 0..k {
                    for l in j..k {
                        meat[j * k + l] += row[j] * row[l] * w;
                    }
                }
            }
            for j in 0..k {
                for l in (j + 1)..k {
                    meat[l * k + j] = meat[j * k + l];
                }
            }
            let tmp = matmul(xtx_inv, &meat, k, k, k);
            matmul(&tmp, xtx_inv, k, k, k)
        }
        _ => {
            // iid: sigma² (X'X)^{-1} — caller handles this case
            unreachable!("sandwich_vcov called with unsupported type: {}", vcov_type)
        }
    }
}

/// Full OLS for no-FE case. Accepts individual column arrays.
/// Handles iid, HC0-HC3, and clustered SEs entirely in Rust.
#[pyfunction]
fn rust_ols_nofe<'py>(
    py: Python<'py>,
    y_col: PyReadonlyArray1<'py, f64>,
    x_cols: Vec<PyReadonlyArray1<'py, f64>>,
    x_names: Vec<String>,
    add_intercept: bool,
    cl_cols: Vec<PyReadonlyArray1<'py, i32>>,
    cl_names: Vec<String>,
    vcov_type: String,
    k_adj: bool,
    g_adj: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,   // beta
    Bound<'py, PyArray2<f64>>,   // vcov
    Bound<'py, PyArray1<f64>>,   // residuals
    f64,                          // r2
    f64,                          // r2_adj
    usize,                        // n_obs
    Vec<usize>,                   // n_clusters per cluster dim
    Vec<String>,                  // final x_names (with _cons if intercept)
)> {
    let y_src = y_col.as_slice().unwrap();
    let n = y_src.len();
    let has_cluster = !cl_cols.is_empty();

    // Borrow x column slices
    let x_slices: Vec<&[f64]> = x_cols.iter().map(|a| a.as_slice().unwrap()).collect();
    let cl_slices: Vec<&[i32]> = cl_cols.iter().map(|a| a.as_slice().unwrap()).collect();

    // Total number of regressors (including intercept)
    let k_base = x_slices.len();
    let k = if add_intercept { k_base + 1 } else { k_base };

    // Build X row-major flat
    let mut x_flat: Vec<f64> = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k_base {
            x_flat.push(x_slices[j][i]);
        }
        if add_intercept {
            x_flat.push(1.0);
        }
    }

    // Copy y
    let y_vec: Vec<f64> = y_src.to_vec();

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

    // Check for NaN beta (singular matrix) — after residuals so we get
    // a cleaner error rather than propagating NaN through R² and VCV
    if beta.iter().any(|v| v.is_nan()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Singular matrix"));
    }

    let r2_adj = if (n as f64 - k as f64) > 0.0 {
        1.0 - (1.0 - r2) * (n as f64 - 1.0) / (n as f64 - k as f64)
    } else {
        0.0
    };

    // --- Variance-covariance ---
    let xtx_inv = invert_kxk(&xtx, k);
    let vcov: Vec<f64>;
    let mut cluster_n_groups: Vec<usize> = Vec::new();

    if has_cluster {
        // Clustered SE — reuse existing CGM logic
        let cl_recoded: Vec<(Vec<i32>, usize)> = cl_slices.iter().map(|cl| recode_vec(cl)).collect();
        for (_, g) in &cl_recoded {
            cluster_n_groups.push(*g);
        }

        let d = cl_recoded.len();
        let mut v_total = vec![0.0_f64; k * k];

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
                let g_adj_factor = if g_adj { g as f64 / (g as f64 - 1.0) } else { 1.0 };
                let k_adj_factor = if k_adj { (n as f64 - 1.0) / (n as f64 - k as f64) } else { 1.0 };
                let dfc = g_adj_factor * k_adj_factor;

                let tmp = matmul(&xtx_inv, &meat, k, k, k);
                let term = matmul(&tmp, &xtx_inv, k, k, k);
                for idx in 0..(k * k) {
                    v_total[idx] += sign * dfc * term[idx];
                }
            }
        }
        vcov = v_total;
    } else if vcov_type == "iid" {
        let sigma2 = if k_adj {
            ss_res / (n - k) as f64
        } else {
            ss_res / n as f64
        };
        vcov = xtx_inv.iter().map(|v| v * sigma2).collect();
    } else {
        // HC0, HC1, HC2, HC3 (no FE in this path, so df_abs = 0)
        vcov = sandwich_vcov(&x_flat, &resid, &xtx_inv, n, k, &vcov_type, 0, k_adj);
    }

    // Build final names
    let mut final_names = x_names;
    if add_intercept {
        final_names.push("_cons".to_string());
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
        r2_adj,
        n,
        cluster_n_groups,
        final_names,
    ))
}

// ---------------------------------------------------------------------------
// HAC / Driscoll-Kraay meat matrices
// ---------------------------------------------------------------------------

/// Newey-West HAC meat matrix: Γ₀ + Σⱼ w(j)(Γⱼ + Γⱼ')
/// with Bartlett kernel weights w(j) = 1 - j/(bw+1).
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

    // Build sorted score (row-major flat)
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

    // Γ₀ = S'S (upper triangle, then symmetrize)
    let mut meat = vec![0.0_f64; k * k];
    for i in 0..n {
        let row = &s_flat[i * k..(i + 1) * k];
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
        for i in lag..n {
            let row_cur = &s_flat[i * k..(i + 1) * k];
            let row_lag = &s_flat[(i - lag) * k..(i - lag + 1) * k];
            for j in 0..k {
                for l in 0..k {
                    gamma[j * k + l] += row_cur[j] * row_lag[l];
                }
            }
        }
        // Add w * (Γⱼ + Γⱼ')
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

    // Recode time_ids to contiguous 0..T-1 (sorted by time value)
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

    // Symmetrize
    for j in 0..k {
        for l in (j + 1)..k {
            meat[l * k + j] = meat[j * k + l];
        }
    }

    let result = Array2::from_shape_vec((k, k), meat).unwrap();
    result.into_pyarray(py)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_demean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_clustered_meat, m)?)?;
    m.add_function(wrap_pyfunction!(rust_recode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ols_core, m)?)?;
    m.add_function(wrap_pyfunction!(rust_absorbed_dof, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ols_from_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ols_nofe, m)?)?;
    m.add_function(wrap_pyfunction!(rust_iv2sls, m)?)?;
    m.add_function(wrap_pyfunction!(rust_hac_meat, m)?)?;
    m.add_function(wrap_pyfunction!(rust_dk_meat, m)?)?;
    Ok(())
}
