"""High-performance feature engineering using Polars on large telemetry time-series."""
import polars as pl
from pathlib import Path
from typing import List, Optional


SMART_COLS = [
    "smart_5_raw",    # Reallocated Sectors Count
    "smart_9_raw",    # Power-On Hours
    "smart_187_raw",  # Reported Uncorrectable Errors
    "smart_188_raw",  # Command Timeout
    "smart_194_raw",  # Temperature
    "smart_197_raw",  # Current Pending Sector Count
]


def compute_telemetry_features(
    df: pl.DataFrame,
    prediction_horizon_days: int = 7,
    rolling_window_days: int = 7,
    top_models: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    Transforms daily drive telemetry into a predictive feature table:
    1. Filters top high-volume models (if specified).
    2. Sorts temporally by (serial_number, date).
    3. Computes rolling 7-day mean and delta changes for critical SMART metrics.
    4. Computes forward target: failure within the next `prediction_horizon_days`.
    """
    # 1. Cast and fill missing SMART values with 0 (or previous observation)
    exprs = [
        pl.col("date").cast(pl.Date),
        pl.col("serial_number").cast(pl.String),
        pl.col("model").cast(pl.String),
        pl.col("failure").cast(pl.Int32),
    ]
    for col in SMART_COLS:
        if col in df.columns:
            exprs.append(pl.col(col).cast(pl.Float64).fill_null(0.0))
        else:
            exprs.append(pl.lit(0.0).alias(col))

    processed = df.select(exprs)

    if top_models:
        processed = processed.filter(pl.col("model").is_in(top_models))

    # 2. Sort by serial_number and date
    processed = processed.sort(["serial_number", "date"])

    # 3. Rolling window and delta features per drive
    feature_exprs = []
    for col in ["smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_194_raw"]:
        # Rolling mean over past window
        feature_exprs.append(
            pl.col(col)
            .rolling_mean(window_size=rolling_window_days, min_samples=1)
            .over("serial_number")
            .alias(f"{col}_mean_{rolling_window_days}d")
        )
        # Delta change from 7 days ago (captures rapid sensor degradation)
        feature_exprs.append(
            (pl.col(col) - pl.col(col).shift(rolling_window_days).over("serial_number"))
            .fill_null(0.0)
            .alias(f"{col}_delta_{rolling_window_days}d")
        )

    processed = processed.with_columns(feature_exprs)

    # 4. Construct forward target: Did drive fail within [t+1, t + horizon]?
    # A drive fails within horizon if max(failure) over the next N days is 1
    # We do a reverse rolling max over the horizon
    target_expr = (
        pl.col("failure")
        .shift(-prediction_horizon_days)
        .rolling_max(window_size=prediction_horizon_days, min_samples=1)
        .over("serial_number")
        .fill_null(0)
        .cast(pl.Int32)
        .alias(f"failure_within_{prediction_horizon_days}d")
    )

    processed = processed.with_columns(target_expr)

    # Filter out drives on their post-failure days (drop rows where current failure == 1)
    # The business goal is early prediction on live drives before they fail
    processed = processed.filter(pl.col("failure") == 0)

    return processed
