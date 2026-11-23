"""Synthetic Backblaze-format data generator and telemetry downloader."""
import datetime
import numpy as np
import polars as pl
from pathlib import Path


def generate_benchmark_telemetry(
    num_drives: int = 2000,
    days: int = 60,
    failure_rate: float = 0.02,
    output_path: str = "data/raw/telemetry_sample.parquet",
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generates realistic large-scale SMART telemetry simulating 2,000+ data center drives
    over 60 days (~120,000+ records) with realistic failure degradation patterns.
    """
    np.random.seed(seed)
    start_date = datetime.date(2024, 1, 1)
    models = ["ST4000DM000", "ST12000NM0007", "TOSHIBA MG07ACA14TA", "WDC WUH721414ALE6L4"]

    records = []
    
    # Pre-select which drives will fail and when
    failing_drives = set(np.random.choice(range(num_drives), size=int(num_drives * failure_rate), replace=False))
    failure_days = {d: np.random.randint(20, days) for d in failing_drives}

    for drive_id in range(num_drives):
        serial = f"WDC-{drive_id:06d}"
        model = np.random.choice(models)
        capacity = 12000000000000
        base_hours = np.random.randint(1000, 40000)
        base_temp = np.random.normal(32.0, 3.0)
        
        will_fail = drive_id in failing_drives
        fail_day = failure_days.get(drive_id, 999)

        s5 = 0.0
        s187 = 0.0
        s197 = 0.0

        for day in range(days):
            if will_fail and day > fail_day:
                break  # Drive was removed from fleet after failure

            current_date = start_date + datetime.timedelta(days=day)
            is_failure_day = 1 if (will_fail and day == fail_day) else 0

            # Normal degradation or accelerated degradation leading to failure
            if will_fail and (fail_day - day) <= 10:
                s5 += np.random.exponential(5.0)
                s187 += np.random.poisson(2.0)
                s197 += np.random.poisson(1.5)
                temp = base_temp + np.random.normal(5.0, 2.0)
            else:
                if np.random.rand() < 0.01:
                    s5 += np.random.poisson(0.1)
                temp = base_temp + np.random.normal(0.0, 1.5)

            records.append({
                "date": current_date,
                "serial_number": serial,
                "model": model,
                "capacity_bytes": capacity,
                "failure": is_failure_day,
                "smart_5_raw": float(s5),
                "smart_9_raw": float(base_hours + day * 24),
                "smart_187_raw": float(s187),
                "smart_188_raw": float(0.0),
                "smart_194_raw": float(np.clip(temp, 18.0, 65.0)),
                "smart_197_raw": float(s197),
            })

    df = pl.DataFrame(records)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"Generated benchmark telemetry dataset: {len(df):,} rows saved to {output_path}")
    return df


if __name__ == "__main__":
    generate_benchmark_telemetry()
