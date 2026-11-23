"""Data contract validation using Pandera."""
import pandera.polars as pa
import polars as pl
from typing import Tuple


RawTelemetrySchema = pa.DataFrameSchema(
    columns={
        "date": pa.Column(pl.Date, nullable=False),
        "serial_number": pa.Column(pl.String, nullable=False),
        "model": pa.Column(pl.String, nullable=False),
        "capacity_bytes": pa.Column(pl.Int64, checks=pa.Check.ge(0), nullable=True),
        "failure": pa.Column(pl.Int32, checks=pa.Check.isin([0, 1]), nullable=False),
        "smart_5_raw": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "smart_9_raw": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "smart_187_raw": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "smart_188_raw": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "smart_194_raw": pa.Column(pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(100.0)], nullable=True),
        "smart_197_raw": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
    },
    coerce=True,
    strict=False,
)


def validate_raw_telemetry(df: pl.DataFrame) -> Tuple[bool, str]:
    """Validates incoming telemetry against raw schema contract."""
    try:
        RawTelemetrySchema.validate(df)
        return True, "Validation successful"
    except Exception as e:
        return False, str(e)
