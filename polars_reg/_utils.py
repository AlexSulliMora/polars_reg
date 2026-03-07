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
        iv_cols = [
            df_clean[c].to_numpy(allow_copy=False).astype(np.float64) for c in spec.instruments
        ]
        instruments = np.column_stack(iv_cols)
        instrument_names = list(spec.instruments)

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
    )
