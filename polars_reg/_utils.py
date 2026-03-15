from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from polars_reg._formula import FormulaSpec

try:
    from polars_reg._native import rust_recode as _rust_recode

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


def _to_codes(series: pl.Series) -> np.ndarray:
    """Convert a Polars Series to contiguous integer group codes (0..G-1).

    Uses Rust native extension when available for integer types.
    Falls back to Polars categorical for strings and other types.
    """
    dtype = series.dtype
    if dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        arr = series.to_numpy().astype(np.int64)
        if _HAS_NATIVE:
            codes, _ = _rust_recode(arr)
            return codes
        # Pure Python fallback
        mn = arr.min()
        mx = arr.max()
        rng = mx - mn
        if rng < 2 * len(arr):
            lut = np.full(rng + 1, -1, dtype=np.int32)
            uniq_shifted = np.unique(arr - mn)
            lut[uniq_shifted] = np.arange(len(uniq_shifted), dtype=np.int32)
            return lut[arr - mn]
        _, codes = np.unique(arr, return_inverse=True)
        return codes.astype(np.int32)
    # String/other types: use Polars categorical encoding
    codes = series.cast(pl.Utf8).cast(pl.Categorical).to_physical().to_numpy()
    return codes.astype(np.int32)


def ensure_polars(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Validate that data is a Polars DataFrame or LazyFrame.

    This is a Polars-native package. Pandas DataFrames are not accepted —
    use ``pl.from_pandas(df)`` before calling any estimator.
    """
    if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(
            f"Expected Polars DataFrame or LazyFrame, got {type(data).__name__}. "
            "Use pl.from_pandas(df) to convert pandas DataFrames."
        )
    return data


def validate_vcov(vcov: str, supported: set[str], model_type: str) -> None:
    """Raise ValueError if vcov is not in the supported set."""
    if vcov not in supported:
        raise ValueError(
            f"vcov={vcov!r} is not supported for {model_type}. "
            f"Available: {', '.join(sorted(supported))}"
        )


def sanitize_inf(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Convert inf/-inf and NaN to null in float columns for uniform null-drop.

    IEEE NaN passes through Polars drop_nulls() unchanged (NaN != null),
    so must be converted to null first. Without this, NaN propagates
    silently through all downstream computation.
    """
    float_cols = [c for c in cols if df[c].dtype.is_float()]
    if float_cols:
        df = df.with_columns(
            [
                pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in float_cols
            ]
        )
    return df


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
    time_array: np.ndarray | None = None
    weights: np.ndarray | None = None


def extract_arrays(
    df: pl.DataFrame | pl.LazyFrame,
    spec: FormulaSpec,
    cluster: list[str] | None = None,
    time: str | None = None,
    weights: str | None = None,
) -> ExtractedArrays:
    """Extract NumPy arrays from a Polars DataFrame given a FormulaSpec."""
    # Determine all columns needed (expand interaction terms to constituent columns)
    exog_cols: list[str] = []
    for col in spec.exog:
        if ":" in col:
            exog_cols.extend(col.split(":"))
        else:
            exog_cols.append(col)
    # Indicator columns are already in exog_cols (without i. prefix)
    all_cols = [spec.depvar] + exog_cols + spec.fe + spec.endog + spec.instruments
    if cluster:
        all_cols += [c for c in cluster if c not in all_cols]
    if time and time not in all_cols:
        all_cols.append(time)
    if weights and weights not in all_cols:
        all_cols.append(weights)
    all_cols = list(dict.fromkeys(all_cols))  # dedupe preserving order

    # Early guard: reject empty DataFrames before any processing
    if isinstance(df, pl.LazyFrame):
        row_count = df.select(pl.len()).collect().item()
    else:
        row_count = len(df)
    if row_count == 0:
        raise ValueError("DataFrame has no observations")

    # Push column selection into LazyFrame before collecting (avoids materializing unused columns)
    if isinstance(df, pl.LazyFrame):
        df = df.select(all_cols).collect()
    else:
        df = df.select(all_cols)

    # Drop rows with nulls in numeric columns (exclude indicator cols from null check)
    numeric_cols = [
        c
        for c in ([spec.depvar] + exog_cols + spec.endog + spec.instruments)
        if c not in spec.indicators
    ]
    if weights:
        numeric_cols.append(weights)
    # Include FE, cluster, and time columns in null-drop
    if spec.fe:
        numeric_cols.extend(spec.fe)
    if cluster:
        numeric_cols.extend(cluster)
    if time:
        numeric_cols.append(time)
    numeric_cols = list(dict.fromkeys(numeric_cols))  # dedupe preserving order

    # Convert IEEE NaN and inf/-inf to Polars null (they pass through drop_nulls silently)
    float_cols = [c for c in numeric_cols if df[c].dtype.is_float()]
    if float_cols:
        df = df.with_columns(
            [
                pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in float_cols
            ]
        )

    df_clean = df.drop_nulls(subset=numeric_cols)

    if len(df_clean) == 0:
        raise ValueError(
            "No observations remain after dropping nulls. "
            "Check for missing data in columns: " + ", ".join(numeric_cols)
        )

    n_obs = len(df_clean)

    # Extract y
    y = df_clean[spec.depvar].to_numpy().astype(np.float64)

    # Pre-compute indicator dummy matrices (sorted levels, drop first as reference)
    _indicator_cache: dict[str, tuple[list[str], np.ndarray]] = {}
    for ind_col in spec.indicators:
        series = df_clean[ind_col]
        levels = sorted(v for v in series.unique().to_list() if v is not None)
        if len(levels) < 2:
            raise ValueError(f"Indicator variable '{ind_col}' has fewer than 2 levels")
        # Drop first level (reference category)
        kept = levels[1:]
        raw = series.to_numpy()
        dummies = np.column_stack([(raw == lv).astype(np.float64) for lv in kept])
        dummy_names = [f"{ind_col}={lv}" for lv in kept]
        _indicator_cache[ind_col] = (dummy_names, dummies)

    # Extract X with optional intercept
    names: list[str] = []
    x_cols: list[np.ndarray] = []
    for col in spec.exog:
        if ":" in col:
            # Interaction term: elementwise product of constituent columns
            # Each part may be an indicator (expand) or continuous (single col)
            parts = col.split(":")
            # Build list of (names, arrays) for each part
            part_expansions: list[list[tuple[str, np.ndarray]]] = []
            for p in parts:
                if p in _indicator_cache:
                    ind_names, ind_arr = _indicator_cache[p]
                    part_expansions.append([(nm, ind_arr[:, j]) for j, nm in enumerate(ind_names)])
                else:
                    part_expansions.append([(p, df_clean[p].to_numpy().astype(np.float64))])
            # Cartesian product of all parts
            combos = part_expansions[0]
            for pe in part_expansions[1:]:
                new_combos = []
                for n1, a1 in combos:
                    for n2, a2 in pe:
                        new_combos.append((f"{n1}:{n2}", a1 * a2))
                combos = new_combos
            for nm, arr in combos:
                names.append(nm)
                x_cols.append(arr)
        elif col in _indicator_cache:
            ind_names, ind_arr = _indicator_cache[col]
            for j, nm in enumerate(ind_names):
                names.append(nm)
                x_cols.append(ind_arr[:, j])
        else:
            x_cols.append(df_clean[col].to_numpy().astype(np.float64))
            names.append(col)
    if spec.add_intercept:
        x_cols.append(np.ones(n_obs, dtype=np.float64))
        names.append("_cons")
    X = np.column_stack(x_cols) if x_cols else np.empty((n_obs, 0), dtype=np.float64)

    # Extract FE as integer codes
    fe_arrays: dict[str, np.ndarray] = {}
    for col in spec.fe:
        fe_arrays[col] = _to_codes(df_clean[col])

    # Extract cluster codes
    cluster_arrays: dict[str, np.ndarray] = {}
    if cluster:
        for col in cluster:
            if col in fe_arrays:
                # Reuse FE codes if same column (avoids redundant conversion)
                cluster_arrays[col] = fe_arrays[col]
            else:
                cluster_arrays[col] = _to_codes(df_clean[col])

    # Extract endogenous and instruments
    endog = None
    instruments = None
    endog_names = None
    instrument_names = None
    if spec.endog:
        endog_cols = [df_clean[c].to_numpy().astype(np.float64) for c in spec.endog]
        endog = np.column_stack(endog_cols)
        endog_names = list(spec.endog)
    if spec.instruments:
        iv_cols = [df_clean[c].to_numpy().astype(np.float64) for c in spec.instruments]
        instruments = np.column_stack(iv_cols)
        instrument_names = list(spec.instruments)

    # Extract time array (numeric, preserving ordering)
    time_array = None
    if time:
        time_array = df_clean[time].cast(pl.Float64).to_numpy()

    # Extract weights
    w = None
    if weights:
        w = df_clean[weights].to_numpy().astype(np.float64)

    return ExtractedArrays(
        y=y,
        X=X,
        names=names,
        n_obs=n_obs,
        fe_arrays=fe_arrays,
        cluster_arrays=cluster_arrays,
        endog=endog,
        instruments=instruments,
        endog_names=endog_names,
        instrument_names=instrument_names,
        time_array=time_array,
        weights=w,
    )
