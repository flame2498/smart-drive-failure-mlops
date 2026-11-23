"""Unit and integration tests for data, feature engineering, and serving API."""
import datetime
import polars as pl
import pytest
from fastapi.testclient import TestClient

from src.data.features import compute_telemetry_features
from src.data.validate import validate_raw_telemetry
from src.serving.api import app


@pytest.fixture
def sample_raw_telemetry():
    return pl.DataFrame({
        "date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        "serial_number": ["DRIVE-001", "DRIVE-001"],
        "model": ["ST12000NM0007", "ST12000NM0007"],
        "capacity_bytes": [12000000000000, 12000000000000],
        "failure": [0, 0],
        "smart_5_raw": [0.0, 1.0],
        "smart_9_raw": [1000.0, 1024.0],
        "smart_187_raw": [0.0, 0.0],
        "smart_188_raw": [0.0, 0.0],
        "smart_194_raw": [30.0, 31.0],
        "smart_197_raw": [0.0, 0.0],
    })


def test_data_validation(sample_raw_telemetry):
    is_valid, msg = validate_raw_telemetry(sample_raw_telemetry)
    assert is_valid, f"Validation failed with: {msg}"


def test_feature_engineering_deltas(sample_raw_telemetry):
    feat_df = compute_telemetry_features(
        sample_raw_telemetry,
        prediction_horizon_days=7,
        rolling_window_days=7,
    )
    assert "smart_5_raw_delta_7d" in feat_df.columns
    assert "failure_within_7d" in feat_df.columns
    assert len(feat_df) == 2


def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_predict_single_endpoint():
    client = TestClient(app)
    payload = {
        "serial_number": "TEST-DRIVE-999",
        "model": "ST12000NM0007",
        "smart_5_raw": 150.0,
        "smart_9_raw": 35000.0,
        "smart_187_raw": 12.0,
        "smart_188_raw": 0.0,
        "smart_194_raw": 42.0,
        "smart_197_raw": 8.0,
        "smart_5_raw_mean_7d": 120.0,
        "smart_5_raw_delta_7d": 30.0,
        "smart_187_raw_mean_7d": 10.0,
        "smart_187_raw_delta_7d": 2.0,
        "smart_197_raw_mean_7d": 6.0,
        "smart_197_raw_delta_7d": 2.0,
        "smart_194_raw_mean_7d": 40.0,
        "smart_194_raw_delta_7d": 2.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_json = response.json()
    assert "failure_risk_score" in res_json
    assert res_json["serial_number"] == "TEST-DRIVE-999"
    assert res_json["risk_level"] in ["CRITICAL", "WARNING", "HEALTHY"]


def test_drift_detection():
    from src.monitoring.drift import detect_sensor_drift
    res = detect_sensor_drift("data/reference/reference_features.parquet")
    assert "drift_detected" in res
    assert res["drift_detected"] is False
    assert res["number_of_columns"] > 0

