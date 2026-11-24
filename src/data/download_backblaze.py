"""Automated Backblaze SMART Hard Drive Telemetry Ingestion and Streaming ETL.

Downloads quarterly Backblaze telemetry archives and transforms them into
optimized columnar Parquet files using memory-efficient Polars streaming.
"""

import argparse
from datetime import datetime
import io
import os
from pathlib import Path
import sys
import time
from typing import List, Optional
import urllib.request
import zipfile

import polars as pl
import yaml

from src.data.validate import validate_raw_telemetry

BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data"

CORE_COLUMNS = [
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
]

SMART_COLUMNS = [
    "smart_5_raw",    # Reallocated Sectors Count
    "smart_9_raw",    # Power-On Hours
    "smart_187_raw",  # Reported Uncorrectable Errors
    "smart_188_raw",  # Command Timeout
    "smart_194_raw",  # Temperature (Celsius)
    "smart_197_raw",  # Current Pending Sector Count
]


def download_quarterly_archive(
    quarter: str = "Q1",
    year: int = 2024,
    dest_dir: str = "data/raw",
) -> Path:
    """Downloads the Backblaze quarterly zip archive if not already present."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    archive_name = f"data_{quarter}_{year}.zip"
    archive_path = dest_path / archive_name
    url = f"{BASE_URL}/{archive_name}"

    if archive_path.exists() and archive_path.stat().st_size > 10_000_000:
        print(f"Archive {archive_name} already exists ({archive_path.stat().st_size / (1024 * 1024):.1f} MB). Skipping download.")
        return archive_path

    print(f"Downloading {url} to {archive_path}...")
    start_time = time.time()

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(100.0, count * block_size * 100.0 / total_size)
            mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            if count % 2000 == 0 or pct >= 100.0:
                elapsed = time.time() - start_time
                speed = mb / max(elapsed, 0.1)
                print(f"  Downloaded {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s", end="\r")

    urllib.request.urlretrieve(url, archive_path, reporthook=_progress)
    print(f"\nDownload completed in {time.time() - start_time:.1f}s: {archive_path}")
    return archive_path


def process_backblaze_archive(
    archive_path: Path,
    output_parquet: str = "data/raw/backblaze_telemetry.parquet",
    top_models: Optional[List[str]] = None,
    max_days: Optional[int] = None,
) -> pl.DataFrame:
    """
    Streams and processes daily telemetry CSV files directly from the zip archive
    without extracting uncompressed gigabytes to disk.
    """
    print(f"Processing archive: {archive_path}")
    all_needed_cols = CORE_COLUMNS + SMART_COLUMNS

    daily_dfs = []
    total_failures = 0
    total_rows = 0

    with zipfile.ZipFile(archive_path, "r") as zf:
        # Find all daily CSV files (ignore macOS __MACOSX metadata)
        csv_names = sorted([
            name for name in zf.namelist()
            if name.endswith(".csv") and not name.startswith("__MACOSX")
        ])

        if max_days:
            csv_names = csv_names[:max_days]

        print(f"Found {len(csv_names)} daily CSV files to process.")

        for idx, csv_name in enumerate(csv_names, 1):
            with zf.open(csv_name) as csv_file:
                # Read CSV header first to identify available columns
                first_line = csv_file.readline().decode("utf-8").strip()
                headers = [h.strip() for h in first_line.split(",")]
                available_cols = [c for c in all_needed_cols if c in headers]

                # Seek back to start
                csv_file.seek(0)
                content = csv_file.read()

                try:
                    df_day = pl.read_csv(
                        io.BytesIO(content),
                        columns=available_cols,
                        ignore_errors=True,
                    )
                except Exception as e:
                    print(f"Warning reading {csv_name}: {e}")
                    continue

                # Filter top models early to reduce memory
                if top_models and "model" in df_day.columns:
                    df_day = df_day.filter(pl.col("model").is_in(top_models))

                if len(df_day) == 0:
                    continue

                date_expr = (
                    pl.col("date").str.to_date()
                    if df_day.schema["date"] == pl.String
                    else pl.col("date").cast(pl.Date)
                )

                # Ensure schema consistency
                schema_exprs = [
                    date_expr,
                    pl.col("serial_number").cast(pl.String),
                    pl.col("model").cast(pl.String),
                    pl.col("capacity_bytes").cast(pl.Int64, strict=False).fill_null(0),
                    pl.col("failure").cast(pl.Int32, strict=False).fill_null(0),
                ]

                for smart_col in SMART_COLUMNS:
                    if smart_col in df_day.columns:
                        col_expr = pl.col(smart_col).cast(pl.Float64, strict=False).fill_null(0.0)
                        # Clean temperature if composite sensor raw format
                        if smart_col == "smart_194_raw":
                            col_expr = (
                                pl.when(col_expr > 100.0)
                                .then(col_expr % 100.0)
                                .otherwise(col_expr)
                                .clip(0.0, 100.0)
                            )
                        schema_exprs.append(col_expr)
                    else:
                        schema_exprs.append(pl.lit(0.0).alias(smart_col))

                df_clean = df_day.select(schema_exprs)
                day_failures = df_clean.filter(pl.col("failure") == 1).shape[0]
                total_failures += day_failures
                total_rows += len(df_clean)
                daily_dfs.append(df_clean)

                if idx % 10 == 0 or idx == len(csv_names):
                    print(
                        f"  Processed {idx}/{len(csv_names)} days: "
                        f"{total_rows:,} rows, {total_failures} failures captured."
                    )

    print("Concatenating processed daily frames...")
    combined_df = pl.concat(daily_dfs, how="vertical")
    # Sort temporally
    combined_df = combined_df.sort(["serial_number", "date"])

    # Validate against Pandera schema contract
    print("Validating dataset against Pandera data contract...")
    is_valid, val_msg = validate_raw_telemetry(combined_df.head(1000))
    if not is_valid:
        print(f"Schema Warning: {val_msg}")
    else:
        print("Pandera contract validation passed successfully.")

    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.write_parquet(out_path, compression="zstd")

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nFinal Parquet saved to: {out_path} ({file_size_mb:.2f} MB)")
    print(f"Summary: {len(combined_df):,} total rows, {combined_df['serial_number'].n_unique():,} unique drives, {total_failures} failure events.")
    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Backblaze Telemetry Ingestion Pipeline")
    parser.add_argument("--quarter", type=str, default="Q1", help="Quarter (e.g. Q1, Q2, Q3, Q4)")
    parser.add_argument("--year", type=int, default=2024, help="Year (e.g. 2024, 2023)")
    parser.add_argument("--sample-days", type=int, default=None, help="Process only first N days for testing")
    parser.add_argument("--output", type=str, default="data/raw/backblaze_telemetry.parquet", help="Output Parquet path")
    parser.add_argument("--config", type=str, default="params.yaml", help="Path to params.yaml")

    args = parser.parse_args()

    top_models = None
    if Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
            top_models = cfg.get("data", {}).get("top_models")

    archive_path = download_quarterly_archive(
        quarter=args.quarter,
        year=args.year,
        dest_dir="data/raw",
    )

    process_backblaze_archive(
        archive_path=archive_path,
        output_parquet=args.output,
        top_models=top_models,
        max_days=args.sample_days,
    )


if __name__ == "__main__":
    main()
