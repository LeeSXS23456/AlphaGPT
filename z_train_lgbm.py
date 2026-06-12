#!/usr/bin/env python3
"""LightGBM rolling-window training on alpha factors.

Purged walk-forward cross-validation:
    |<-- train (2yr) -->|<-purge(22d)->|<-test(1mo)->|
    |  inner 80% |val 20% |

- Validation for early stopping is drawn from within the training period.
- Purge gap prevents factor lookback (max 20d) from overlapping train→test.
- ``copy.deepcopy`` on all splits to prevent reference-based leakage.
- Cross-year data loading when windows span year boundaries.

Metrics: IC, Rank IC, decile long-short returns, Sharpe, max drawdown, hit rate.
"""

from __future__ import annotations

import copy
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from z_build_dataset import load_dataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "factor_a1c",
    "factor_a2",
    "factor_b1",
    "factor_c1",
    "factor_c1_v2",
    "factor_c2_v2",
]
TARGET_COL = "fwd_return"

# Rolling window (in months)
TRAIN_MONTHS = 24
VAL_RATIO = 0.20          # fraction of training period held out for early stopping
PURGE_TRADING_DAYS = 22   # max factor lookback (20d) + forward reach (2d)
TEST_MONTHS = 1
STEP_MONTHS = 1            # roll forward 1 month per iteration


@dataclass
class WindowResult:
    """Metrics from a single rolling window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_val: int
    n_test: int

    # Out-of-sample
    ic_mean: float
    ic_std: float
    ic_ir: float
    rank_ic_mean: float
    rank_ic_std: float
    ls_sharpe: float          # decile long-short Sharpe (annualized)
    ls_max_dd: float           # decile long-short max drawdown
    hit_rate: float            # fraction of days with positive IC

    # In-sample (on validation set)
    ic_is: float

    # Feature importance (gain-based, dict of factor → importance)
    feature_gains: dict[str, float]


# ---------------------------------------------------------------------------
# LightGBM parameters
# ---------------------------------------------------------------------------

LGB_PARAMS = {
    "objective": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 8,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 1000,
    "min_sum_hessian_in_leaf": 1e-3,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": 0,
    "num_threads": 16,
    "seed": 42,
}

LGB_FIT_PARAMS = {
    "num_boost_round": 2000,
    "early_stopping_rounds": 50,
}


# ---------------------------------------------------------------------------
# Data loading (cross-year aware)
# ---------------------------------------------------------------------------


def _load_date_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Load data for a date range, handling year boundaries."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    years_needed = set()
    for yr in range(start.year, end.year + 1):
        years_needed.add(yr)
    # Also check if dates span across the year boundary
    years_needed.update([start.year, end.year])

    frames = []
    for yr in sorted(years_needed):
        try:
            yr_df = load_dataset(yr)
            yr_df["date"] = pd.to_datetime(yr_df["date"])
            frames.append(yr_df)
        except FileNotFoundError:
            continue

    if not frames:
        raise FileNotFoundError(f"No data for years {years_needed}")

    df = pd.concat(frames, ignore_index=True)
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Train / val / test splitting with purge
# ---------------------------------------------------------------------------


def _purge_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    purge_days: int,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, val, test with purge gap.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset, sorted by date.
    train_end : pd.Timestamp
        Last date included in the training period.
    purge_days : int
        Number of TRADING days to purge between train and test.
    val_ratio : float
        Fraction of training period held out for validation.

    Returns
    -------
    train, val, test : (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Deep-copied splits.  Test starts after ``train_end + purge_days``
        trading days.  Val is the last ``val_ratio`` of training dates.
    """
    dates_sorted = sorted(df["date"].unique())
    dates_series = pd.Series(dates_sorted)

    # Find train_end index
    train_end_idx = dates_series.searchsorted(train_end)
    if isinstance(train_end_idx, np.ndarray):
        train_end_idx = train_end_idx[0]
    train_end_date = dates_sorted[min(train_end_idx, len(dates_sorted) - 1)]

    # Test start: train_end + purge_days trading days
    test_start_idx = min(train_end_idx + purge_days + 1, len(dates_sorted) - 1)
    test_start_date = dates_sorted[test_start_idx]

    # Take 21 * TEST_MONTHS trading days, minus 1 to avoid overlap with the
    # next window's test_start (consistent with _generate_windows).
    n_test_days = 21 * TEST_MONTHS - 1
    test_last_idx = min(test_start_idx + n_test_days, len(dates_sorted) - 1)
    test_end_date = dates_sorted[test_last_idx]

    # --- Train set: up to train_end_date ---
    train_mask = df["date"] <= train_end_date
    train_df = df[train_mask]

    # --- Split train into inner train and val ---
    train_dates = sorted(train_df["date"].unique())
    n_val_dates = max(1, int(len(train_dates) * val_ratio))
    val_dates = set(train_dates[-n_val_dates:])
    inner_train_dates = set(train_dates[:-n_val_dates])

    inner_train_df = train_df[train_df["date"].isin(inner_train_dates)]
    val_df = train_df[train_df["date"].isin(val_dates)]

    # --- Test set: from test_start_date to test_end_date ---
    test_mask = (df["date"] >= test_start_date) & (df["date"] <= test_end_date)
    test_df = df[test_mask]

    return (
        copy.deepcopy(inner_train_df),
        copy.deepcopy(val_df),
        copy.deepcopy(test_df),
    )


# ---------------------------------------------------------------------------
# Custom eval metric — Rank IC for monitoring & early stopping
# ---------------------------------------------------------------------------


def _rank_ic_metric(preds: np.ndarray, train_data: lgb.Dataset) -> tuple[str, float, bool]:
    """Rank IC (Spearman) on the validation set.  Higher is better."""
    y = train_data.get_label()
    ic = float(pd.Series(preds).corr(pd.Series(y), method="spearman"))
    return "rank_ic", ic, True


# ---------------------------------------------------------------------------
# Single-window training
# ---------------------------------------------------------------------------


def _train_one_window(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[lgb.Booster, dict[str, float], float, int]:
    """Train a LightGBM model on one window.

    Returns
    -------
    model, feature_gains, val_ic, best_iteration
    """
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params=LGB_PARAMS,
        train_set=dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        feval=_rank_ic_metric,
        num_boost_round=LGB_FIT_PARAMS["num_boost_round"],
        callbacks=[
            lgb.early_stopping(LGB_FIT_PARAMS["early_stopping_rounds"]),
            lgb.log_evaluation(period=50),
        ],
    )

    # Feature importance (gain-based)
    gain = model.feature_importance(importance_type="gain")
    feature_gains = dict(zip(FEATURE_COLS, gain))
    # Normalize
    total = sum(feature_gains.values())
    if total > 0:
        feature_gains = {k: v / total for k, v in feature_gains.items()}

    # Validation IC
    y_pred_val = model.predict(X_val)
    val_ic = float(np.corrcoef(y_pred_val, y_val)[0, 1]) if len(y_val) > 1 else 0.0

    best_iter = model.best_iteration

    return model, feature_gains, val_ic, best_iter


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
) -> dict:
    """Compute all evaluation metrics from predictions.

    Long-short is computed as top decile minus bottom decile, cross-sectionally
    per date, then aggregated.  Sign-aligned per our standard approach.

    Parameters
    ----------
    y_true : np.ndarray
        Actual forward returns.
    y_pred : np.ndarray
        Model predictions.
    dates : np.ndarray
        Date for each observation (for cross-sectional grouping).

    Returns
    -------
    dict with ic_mean, ic_std, ic_ir, rank_ic_mean, rank_ic_std,
         ls_sharpe, ls_max_dd, hit_rate.
    """
    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    unique_dates = sorted(df["date"].unique())

    ic_list = []
    rank_ic_list = []
    ls_returns = []

    for d in unique_dates:
        day = df[df["date"] == d]
        if len(day) < 50:
            continue

        # IC
        ic = day["y_pred"].corr(day["y_true"])
        if pd.notna(ic):
            ic_list.append(ic)

        # Rank IC
        ric = day["y_pred"].corr(day["y_true"], method="spearman")
        if pd.notna(ric):
            rank_ic_list.append(ric)

        # Decile long-short
        try:
            day["decile"] = pd.qcut(day["y_pred"], 10, labels=False, duplicates="drop")
            if day["decile"].nunique() >= 2:
                top = day.loc[day["decile"] == day["decile"].max(), "y_true"].mean()
                bot = day.loc[day["decile"] == day["decile"].min(), "y_true"].mean()
                ls_returns.append(top - bot)
        except Exception:
            continue

    ic_arr = np.array(ic_list)
    rank_ic_arr = np.array(rank_ic_list)
    ls_arr = np.array(ls_returns)

    # Sign-align long-short
    mean_ic = float(np.nanmean(ic_arr)) if len(ic_arr) > 0 else 0.0
    if mean_ic < 0:
        ls_arr = -ls_arr

    ls_mean = float(np.nanmean(ls_arr)) if len(ls_arr) > 0 else 0.0
    ls_std = float(np.nanstd(ls_arr, ddof=1)) if len(ls_arr) > 1 else 0.0
    ls_sharpe = (ls_mean / ls_std) * np.sqrt(252) if ls_std > 0 else 0.0

    # Max drawdown
    if len(ls_arr) > 0:
        cum = np.cumprod(1 + ls_arr)
        running_max = np.maximum.accumulate(cum)
        drawdowns = cum / running_max - 1
        ls_max_dd = float(np.min(drawdowns))
    else:
        ls_max_dd = 0.0

    ic_ir = float(np.nanmean(ic_arr) / np.nanstd(ic_arr, ddof=1)) if (
        len(ic_arr) > 1 and np.nanstd(ic_arr, ddof=1) > 0
    ) else 0.0

    hit_rate = float(np.nanmean(ic_arr > 0)) if len(ic_arr) > 0 else 0.0

    return {
        "ic_mean": mean_ic,
        "ic_std": float(np.nanstd(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0,
        "ic_ir": ic_ir,
        "rank_ic_mean": float(np.nanmean(rank_ic_arr)) if len(rank_ic_arr) > 0 else 0.0,
        "rank_ic_std": float(np.nanstd(rank_ic_arr, ddof=1)) if len(rank_ic_arr) > 1 else 0.0,
        "ls_sharpe": ls_sharpe,
        "ls_max_dd": ls_max_dd,
        "hit_rate": hit_rate,
    }


# ---------------------------------------------------------------------------
# Generate window boundaries
# ---------------------------------------------------------------------------


def _generate_windows(df: pd.DataFrame) -> list[dict]:
    """Generate rolling window (train_start, train_end, test_start, test_end)."""
    dates_sorted = sorted(df["date"].unique())
    dates_series = pd.Series(dates_sorted)

    # Trading days per month (approximate)
    tpm = 21

    windows = []
    # Start from the earliest possible: TRAIN_MONTHS into the data
    start_idx = TRAIN_MONTHS * tpm

    for test_start_idx in range(start_idx, len(dates_sorted) - tpm, STEP_MONTHS * tpm):
        train_end_idx = test_start_idx - PURGE_TRADING_DAYS - 1
        if train_end_idx < tpm:  # need at least 1 month of training
            continue

        train_start_idx = max(0, train_end_idx - TRAIN_MONTHS * tpm)
        test_end_idx = min(test_start_idx + tpm * TEST_MONTHS - 1, len(dates_sorted) - 1)

        windows.append({
            "train_start": dates_sorted[train_start_idx],
            "train_end": dates_sorted[train_end_idx],
            "test_start": dates_sorted[test_start_idx],
            "test_end": dates_sorted[test_end_idx],
        })

    return windows


# ---------------------------------------------------------------------------
# Main rolling loop
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 78)
    print("  LightGBM Rolling Training — Purged Walk-Forward")
    print("=" * 78)
    print(f"  Train: {TRAIN_MONTHS}mo  |  Purge: {PURGE_TRADING_DAYS}d  |  "
          f"Test: {TEST_MONTHS}mo  |  Step: {STEP_MONTHS}mo")
    print(f"  Features: {len(FEATURE_COLS)}  |  Val ratio: {VAL_RATIO:.0%}")
    print(f"\n  LightGBM Config:")
    for k, v in LGB_PARAMS.items():
        if k != "verbosity":
            print(f"    {k}: {v}")
    print(f"    num_boost_round: {LGB_FIT_PARAMS['num_boost_round']}  "
          f"(early_stopping: {LGB_FIT_PARAMS['early_stopping_rounds']})")

    # --- Load full dataset --------------------------------------------------
    t0 = time.perf_counter()
    print("\n[1/4] Loading dataset ...", end=" ", flush=True)
    df = load_dataset()
    df["date"] = pd.to_datetime(df["date"])
    # # TO DEBUG
    # df = df[df["date"] >= pd.to_datetime("2024-01-01")]
    df = df.sort_values(["date", "stock_id"]).reset_index(drop=True)
    print(f"done ({time.perf_counter() - t0:.1f}s)")
    print(f"      {len(df):,} rows, {df['date'].nunique():,} trading days")

    # --- Generate windows ---------------------------------------------------
    windows = _generate_windows(df)
    print(f"\n[2/4] Generated {len(windows)} rolling windows")
    if windows:
        w = windows[0]
        print(f"      First: train [{w['train_start'].date()} → {w['train_end'].date()}]  "
              f"test [{w['test_start'].date()} → {w['test_end'].date()}]")
        w = windows[-1]
        print(f"      Last:  train [{w['train_start'].date()} → {w['train_end'].date()}]  "
              f"test [{w['test_start'].date()} → {w['test_end'].date()}]")

    # --- Rolling training ---------------------------------------------------
    print(f"\n[3/4] Rolling training ...")
    results: list[WindowResult] = []
    all_oos_preds: list[pd.DataFrame] = []
    cumulative_gains: dict[str, list[float]] = {f: [] for f in FEATURE_COLS}

    for i, w in enumerate(windows):
        t1 = time.perf_counter()
        train_start_str = str(w["train_start"].date())
        train_end_str = str(w["train_end"].date())
        test_start_str = str(w["test_start"].date())
        test_end_str = str(w["test_end"].date())

        # Load data for this window (handles year boundaries)
        window_df = _load_date_range(train_start_str, test_end_str)

        # Purge split
        train_df, val_df, test_df = _purge_split(
            window_df,
            train_end=w["train_end"],
            purge_days=PURGE_TRADING_DAYS,
        )

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Train
        model, feature_gains, val_ic, best_iter = _train_one_window(train_df, val_df)

        # Predict OOS
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df[TARGET_COL].values
        y_pred = model.predict(X_test)

        # Evaluate OOS
        oos_metrics = _evaluate_predictions(
            y_test, y_pred,
            test_df["date"].values,  # np.array of Timestamps
        )

        # Store OOS predictions
        pred_df = test_df[["date", "stock_id"]].copy()
        pred_df["y_true"] = y_test
        pred_df["y_pred"] = y_pred
        pred_df["window"] = i
        all_oos_preds.append(pred_df)

        # Track feature importance
        for f in FEATURE_COLS:
            cumulative_gains[f].append(feature_gains.get(f, 0.0))

        wr = WindowResult(
            train_start=train_start_str,
            train_end=train_end_str,
            test_start=test_start_str,
            test_end=test_end_str,
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            ic_mean=oos_metrics["ic_mean"],
            ic_std=oos_metrics["ic_std"],
            ic_ir=oos_metrics["ic_ir"],
            rank_ic_mean=oos_metrics["rank_ic_mean"],
            rank_ic_std=oos_metrics["rank_ic_std"],
            ls_sharpe=oos_metrics["ls_sharpe"],
            ls_max_dd=oos_metrics["ls_max_dd"],
            hit_rate=oos_metrics["hit_rate"],
            ic_is=val_ic,
            feature_gains=feature_gains,
        )
        results.append(wr)

        elapsed = time.perf_counter() - t1
        print(f"  [{i+1:3d}/{len(windows)}]  "
              f"test {test_start_str} → {test_end_str}  "
              f"n_train={len(train_df):,}  n_val={len(val_df):,}  n_test={len(test_df):,}  "
              f"trees={best_iter:4d}  "
              f"IC={oos_metrics['ic_mean']:+.4f}  "
              f"RIC={oos_metrics['rank_ic_mean']:+.4f}  "
              f"Sharpe={oos_metrics['ls_sharpe']:+.2f}  "
              f"({elapsed:.1f}s)")

        # Cumulative OOS summary every 20 windows
        if (i + 1) % 20 == 0:
            cum_preds = pd.concat(all_oos_preds, ignore_index=True)
            cum_metrics = _evaluate_predictions(
                cum_preds["y_true"].values,
                cum_preds["y_pred"].values,
                cum_preds["date"].values,
            )
            print(f"  --- cumulative after {i+1} windows --- "
                  f"IC={cum_metrics['ic_mean']:+.4f}  "
                  f"RIC={cum_metrics['rank_ic_mean']:+.4f}  "
                  f"Sharpe={cum_metrics['ls_sharpe']:+.2f}  "
                  f"Hit={cum_metrics['hit_rate']:.1%}")

    print(f"\n  Completed {len(results)} windows.")

    # --- Aggregate results --------------------------------------------------
    print(f"\n[4/4] Aggregate Evaluation")
    print()
    print("=" * 78)
    print("  Out-of-Sample Performance (aggregated across all test windows)")
    print("=" * 78)

    # Combine all OOS predictions for global evaluation
    all_preds = pd.concat(all_oos_preds, ignore_index=True)
    global_metrics = _evaluate_predictions(
        all_preds["y_true"].values,
        all_preds["y_pred"].values,
        all_preds["date"].values,
    )

    print(f"\n  {'OOS IC Mean':<28s} {global_metrics['ic_mean']:>+.4f}")
    print(f"  {'OOS IC Std':<28s} {global_metrics['ic_std']:>.4f}")
    print(f"  {'OOS IC IR':<28s} {global_metrics['ic_ir']:>+.2f}")
    print(f"  {'OOS Rank IC Mean':<28s} {global_metrics['rank_ic_mean']:>+.4f}")
    print(f"  {'OOS Rank IC Std':<28s} {global_metrics['rank_ic_std']:>.4f}")
    print(f"  {'OOS Decile L-S Sharpe':<28s} {global_metrics['ls_sharpe']:>+.2f}")
    print(f"  {'OOS Max Drawdown':<28s} {global_metrics['ls_max_dd']:>+.1%}")
    print(f"  {'OOS Hit Rate':<28s} {global_metrics['hit_rate']:>.1%}")

    # Per-window IC summary
    ic_means = [r.ic_mean for r in results]
    rank_ic_means = [r.rank_ic_mean for r in results]
    sharpes = [r.ls_sharpe for r in results]
    hit_rates = [r.hit_rate for r in results]

    print(f"\n  Per-Window Summary (n={len(results)}):")
    print(f"  {'':<20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'P25':>8s} "
          f"{'P50':>8s} {'P75':>8s} {'Max':>8s}")
    for label, arr in [("IC", ic_means), ("Rank IC", rank_ic_means),
                        ("Sharpe", sharpes), ("Hit Rate", hit_rates)]:
        a = np.array(arr)
        print(f"  {label:<20s} {np.mean(a):>+8.4f} {np.std(a, ddof=1):>8.4f} "
              f"{np.min(a):>+8.4f} {np.percentile(a, 25):>+8.4f} "
              f"{np.percentile(a, 50):>+8.4f} {np.percentile(a, 75):>+8.4f} "
              f"{np.max(a):>+8.4f}")

    # Feature importance (average across windows)
    print(f"\n  Average Feature Importance (gain, normalized):")
    avg_gains = {f: np.mean(cumulative_gains[f]) for f in FEATURE_COLS}
    total = sum(avg_gains.values())
    for f in sorted(FEATURE_COLS, key=lambda x: avg_gains[x], reverse=True):
        pct = avg_gains[f] / total * 100 if total > 0 else 0
        print(f"    {f:<20s}  {pct:5.1f}%")

    # In-sample vs OOS IC
    is_ics = [r.ic_is for r in results]
    print(f"\n  In-Sample vs Out-of-Sample IC:")
    print(f"    IS  (val):  mean = {np.mean(is_ics):+.4f}")
    print(f"    OOS (test): mean = {np.mean(ic_means):+.4f}")

    # --- Save to cache --------------------------------------------------------
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. OOS predictions (one row per stock per day)
    preds_file = cache_dir / "oos_predictions.parquet"
    all_preds.to_parquet(preds_file, index=False)
    print(f"\n  Predictions saved: {preds_file}")
    print(f"    {len(all_preds):,} rows  |  columns: date, stock_id, y_true, y_pred, window")

    # 2. Daily IC / Rank IC (one row per date)
    daily_file = cache_dir / "daily_metrics.csv"
    daily_rows = []
    for d, grp in all_preds.groupby("date"):
        if len(grp) < 50:
            continue
        ic = grp["y_pred"].corr(grp["y_true"])
        ric = grp["y_pred"].corr(grp["y_true"], method="spearman")
        try:
            decile = pd.qcut(grp["y_pred"], 10, labels=False, duplicates="drop")
            if decile.nunique() >= 2:
                top = grp.loc[decile == decile.max(), "y_true"].mean()
                bot = grp.loc[decile == decile.min(), "y_true"].mean()
                ls_ret = top - bot
            else:
                ls_ret = np.nan
        except Exception:
            ls_ret = np.nan
        daily_rows.append({
            "date": d,
            "n_stocks": len(grp),
            "ic": ic if pd.notna(ic) else np.nan,
            "rank_ic": ric if pd.notna(ric) else np.nan,
            "ls_return": ls_ret,
        })
    daily_df = pd.DataFrame(daily_rows).sort_values("date")
    daily_df.to_csv(daily_file, index=False)
    print(f"  Daily metrics saved: {daily_file}")
    print(f"    {len(daily_df):,} dates  |  columns: date, n_stocks, ic, rank_ic, ls_return")

    # 3. Aggregate metrics JSON
    metrics_file = cache_dir / "aggregate_metrics.json"
    metrics_data = {
        "global_oos": global_metrics,
        "per_window_summary": {
            "n_windows": len(results),
            "ic": {
                "mean": float(np.mean(ic_means)),
                "std": float(np.std(ic_means, ddof=1)),
                "min": float(np.min(ic_means)),
                "max": float(np.max(ic_means)),
            },
            "rank_ic": {
                "mean": float(np.mean(rank_ic_means)),
                "std": float(np.std(rank_ic_means, ddof=1)),
                "min": float(np.min(rank_ic_means)),
                "max": float(np.max(rank_ic_means)),
            },
            "sharpe": {
                "mean": float(np.mean(sharpes)),
                "std": float(np.std(sharpes, ddof=1)),
                "min": float(np.min(sharpes)),
                "max": float(np.max(sharpes)),
            },
            "hit_rate": {
                "mean": float(np.mean(hit_rates)),
                "std": float(np.std(hit_rates, ddof=1)),
                "min": float(np.min(hit_rates)),
                "max": float(np.max(hit_rates)),
            },
        },
        "in_sample_vs_oos": {
            "is_val_ic_mean": float(np.mean(is_ics)),
            "oos_test_ic_mean": float(np.mean(ic_means)),
        },
        "feature_importance": {
            f: float(avg_gains[f]) for f in
            sorted(FEATURE_COLS, key=lambda x: avg_gains[x], reverse=True)
        },
        "config": {
            "train_months": TRAIN_MONTHS,
            "purge_trading_days": PURGE_TRADING_DAYS,
            "test_months": TEST_MONTHS,
            "step_months": STEP_MONTHS,
            "val_ratio": VAL_RATIO,
        },
        "lgb_params": {k: v for k, v in LGB_PARAMS.items()
                        if k not in ("verbosity", "num_threads")},
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2, default=str)
    print(f"  Metrics saved:    {metrics_file}")

    # 3. Per-window detail CSV (for debugging individual windows)
    windows_file = cache_dir / "window_results.csv"
    pd.DataFrame([{
        "train_start": r.train_start,
        "train_end": r.train_end,
        "test_start": r.test_start,
        "test_end": r.test_end,
        "n_train": r.n_train,
        "n_val": r.n_val,
        "n_test": r.n_test,
        "ic_oos": r.ic_mean,
        "ic_is": r.ic_is,
        "rank_ic": r.rank_ic_mean,
        "sharpe": r.ls_sharpe,
        "max_dd": r.ls_max_dd,
        "hit_rate": r.hit_rate,
    } for r in results]).to_csv(windows_file, index=False)
    print(f"  Windows saved:    {windows_file}")

    print()
    print("=" * 78)
    print("  Rolling training complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
