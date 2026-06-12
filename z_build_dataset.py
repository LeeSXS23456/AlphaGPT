#!/usr/bin/env python3
"""Build the alpha-factor dataset for LightGBM training.

Computes all 6 working alpha factors + tradeable forward returns for every
stock-date, aligns them via inner join, and saves as yearly parquet files.

CRITICAL: Forward return = (open_{t+2} - open_{t+1}) / open_{t+1}.
Factors are known at day t close.  Earliest trade is t+1 open.
Hold one full day → sell at t+2 open.  Zero lookahead.

Output structure:
    data/processed/alpha_factors/
        2020.parquet
        2021.parquet
        ...
        2026.parquet
        debug.parquet    (first 10 trading days, for fast iteration)

Usage:
    python build_dataset.py              # full build
    python -c "from build_dataset import load_dataset; df = load_dataset(2024)"
    python -c "from build_dataset import load_dataset; df = load_dataset('debug')"
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from alphas.momentum import (
    _dict_to_ohlcv_dataframe,
    compute_a1c,
    compute_a2,
    load_data,
)
from alphas.liquidity import compute_b1
from alphas.microstructure import compute_c1, compute_c1_v2, compute_c2_v2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("data/processed/alpha_factors")
DEBUG_DAYS = 10  # trading days in debug sample

FACTOR_SPECS: list[tuple[str, str, callable]] = [
    ("A1c",    "20d range-adj overnight momentum",  compute_a1c),
    ("A2",     "20d vol-adj overnight momentum",     compute_a2),
    ("B1",     "volume shock",                        compute_b1),
    ("C1",     "range expansion ratio",               compute_c1),
    ("C1_v2",  "log price range",                     compute_c1_v2),
    ("C2_v2",  "CLV × ln(range)",                     compute_c2_v2),
]


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_dataset(year: int | str | None = None) -> pd.DataFrame:
    """Load the alpha-factor dataset.

    Parameters
    ----------
    year : int, str, or None
        - ``int`` (e.g. 2024): load that year's parquet file.
        - ``"debug"``: load the small debug sample.
        - ``None``: load all years and concatenate.

    Returns
    -------
    pd.DataFrame
        Columns: date, stock_id, fwd_return, factor_a1c, ..., factor_c2_v2.
    """
    if year == "debug":
        fp = OUTPUT_DIR / "debug.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Debug dataset not found: {fp}\nRun build_dataset.py first.")
        return pd.read_parquet(fp)

    if year is not None:
        fp = OUTPUT_DIR / f"{year}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Year {year} not found: {fp}")
        return pd.read_parquet(fp)

    # Load all years
    files = sorted(OUTPUT_DIR.glob("20*.parquet"))
    if not files:
        raise FileNotFoundError(f"No yearly parquet files found in {OUTPUT_DIR}")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("  Alpha-Return Dataset Builder")
    print("=" * 72)

    # --- Load data ----------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[1/3] Loading market data ...", end=" ", flush=True)
    data = load_data()
    n_dates = len(data)
    n_obs = sum(len(stocks) for stocks in data.values())
    print(f"done ({time.perf_counter() - t0:.1f}s)")
    print(f"      {n_dates:,} dates, {n_obs:,} stock-day observations")

    # --- Compute factors ----------------------------------------------------
    print("\n[2/3] Computing factors ...")
    factors: dict[str, pd.Series] = {}

    for name, desc, compute_fn in FACTOR_SPECS:
        t0 = time.perf_counter()
        print(f"      {name} ({desc}) ...", end=" ", flush=True)
        factor = compute_fn(data)
        factor.name = f"factor_{name.lower()}"
        factors[name] = factor
        print(f"done ({time.perf_counter() - t0:.1f}s)  —  {len(factor):,} obs")

    # --- Compute forward return (open-to-open, zero lookahead) ---------------
    print("      fwd_return (open-to-open) ...", end=" ", flush=True)
    t0 = time.perf_counter()

    ohlcv = _dict_to_ohlcv_dataframe(data)
    open_ = ohlcv["open"]

    fwd_open_1 = open_.groupby("stock_id").shift(-1)   # t+1 open
    fwd_open_2 = open_.groupby("stock_id").shift(-2)   # t+2 open
    fwd_return = (fwd_open_2 - fwd_open_1) / fwd_open_1.where(fwd_open_1 > 1e-8, other=np.nan)
    fwd_return.name = "fwd_return"
    print(f"done ({time.perf_counter() - t0:.1f}s)  —  {fwd_return.count():,} obs")

    # --- Align and merge ----------------------------------------------------
    print("\n[3/3] Aligning, merging, and saving by year ...")
    t0 = time.perf_counter()

    # Build the full merged DataFrame
    df = fwd_return.reset_index()
    df.columns = ["date", "stock_id", "fwd_return"]

    for name, _, _ in FACTOR_SPECS:
        s = factors[name]
        s_df = s.reset_index()
        s_df.columns = ["date", "stock_id", f"factor_{name.lower()}"]
        df = df.merge(s_df, on=["date", "stock_id"], how="inner")

    df = df.sort_values(["date", "stock_id"]).reset_index(drop=True)
    df = df.dropna()

    # Extract year for partitioning
    df["year"] = pd.to_datetime(df["date"]).dt.year
    years = sorted(df["year"].unique())

    elapsed = time.perf_counter() - t0
    print(f"      Merged: {len(df):,} rows ({elapsed:.1f}s)")

    # --- Save by year -------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for yr in years:
        t1 = time.perf_counter()
        yr_df = df[df["year"] == yr].drop(columns=["year"])
        fp = OUTPUT_DIR / f"{yr}.parquet"
        yr_df.to_parquet(fp, index=False)
        print(f"      {yr}.parquet  —  {len(yr_df):>10,} rows  ({time.perf_counter() - t1:.1f}s)")

    # --- Save debug sample (first N trading days) ----------------------------
    t1 = time.perf_counter()
    dates_unique = sorted(df["date"].unique())
    debug_dates = dates_unique[:DEBUG_DAYS]
    debug_df = df[df["date"].isin(debug_dates)].drop(columns=["year"])
    fp_debug = OUTPUT_DIR / "debug.parquet"
    debug_df.to_parquet(fp_debug, index=False)
    print(f"      debug.parquet  —  {len(debug_df):>10,} rows  "
          f"({debug_dates[0]} → {debug_dates[-1]})  "
          f"({time.perf_counter() - t1:.1f}s)")

    # Drop year column for summary stats
    df = df.drop(columns=["year"])

    # --- Summary ------------------------------------------------------------
    print()
    print("-" * 72)
    print("  Dataset Summary")
    print("-" * 72)
    dates_sorted = sorted(df["date"].unique())
    print(f"  Date range:         {dates_sorted[0]}  →  {dates_sorted[-1]}")
    print(f"  Trading days:       {len(dates_sorted):,}")
    print(f"  Unique stocks:      {df['stock_id'].nunique():,}")
    print(f"  Total rows:         {len(df):,}")
    print(f"  Avg stocks/day:     {len(df) / len(dates_sorted):.0f}")
    print(f"  Output directory:   {OUTPUT_DIR}/")
    print(f"  Partitions:         {len(years)} yearly + 1 debug")

    # Forward return distribution
    fwd = df["fwd_return"]
    print("\n  Forward Return (open_{t+1} → open_{t+2}) Distribution:")
    print(f"    Mean:    {fwd.mean():+.6f}")
    print(f"    Std:     {fwd.std():.6f}")
    print(f"    Min:     {fwd.min():+.6f}")
    print(f"    P1:      {fwd.quantile(0.01):+.6f}")
    print(f"    P99:     {fwd.quantile(0.99):+.6f}")
    print(f"    Max:     {fwd.max():+.6f}")

    # Factor correlation matrix
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    corr = df[factor_cols].corr()
    print(f"\n  Factor Correlation Matrix:")
    print(f"  {'':>14s}", end="")
    for c in factor_cols:
        print(f" {c[7:]:>10s}", end="")
    print()
    for c1 in factor_cols:
        print(f"  {c1[7:]:>14s}", end="")
        for c2 in factor_cols:
            print(f" {corr.loc[c1, c2]:10.3f}", end="")
        print()

    # Factor vs forward return correlation (linear IC check)
    print(f"\n  Factor → fwd_return Correlation:")
    for col in factor_cols:
        ic = df[col].corr(df["fwd_return"])
        print(f"    {col[7:]:>14s}: {ic:+.6f}")

    # Yearly breakdown
    print(f"\n  Yearly Row Counts:")
    for yr in years:
        yr_df = df[pd.to_datetime(df["date"]).dt.year == yr]
        print(f"    {yr}: {len(yr_df):>10,}")

    print()
    print("=" * 72)
    print("  Dataset ready for LightGBM.")
    print(f"  Quick load: load_dataset('debug')  or  load_dataset(2024)")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
