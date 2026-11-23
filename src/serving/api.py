"""FastAPI production inference microservice with health checks and metrics."""
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from prometheus_client import Counter, Histogram, make_asgi_app

from src.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    DriveTelemetryInput,
    PredictionResponse,
)

app = FastAPI(
    title="Cloud Hard Drive Failure Prediction Service",
    description="Low-latency inference API predicting hardware storage failure within 7 days using SMART telemetry.",
    version="1.0.0",
)

# Prometheus metrics
PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total prediction requests received")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency of inference in seconds")

MODEL_PATH = Path("models/lgb_model.joblib")
_model_bundle = None


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train model first.")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


@app.on_event("startup")
def startup_event():
    try:
        get_model_bundle()
    except Exception as e:
        print(f"Warning: {e}")


# Attach Prometheus metrics endpoint at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    bundle = get_model_bundle()
    return {
        "status": "healthy",
        "model_loaded": bundle is not None,
        "features": bundle["feature_cols"],
    }


def _score_drive(input_data: DriveTelemetryInput, model, feature_cols) -> PredictionResponse:
    data_dict = input_data.model_dump()
    row = {col: data_dict.get(col, 0.0) for col in feature_cols}
    df_row = pd.DataFrame([row])

    prob = float(model.predict_proba(df_row)[0, 1])
    is_failing = int(prob >= 0.5)

    if prob >= 0.75:
        risk_level = "CRITICAL"
        action = "Evacuate data immediately and schedule drive replacement within 48h."
    elif prob >= 0.40:
        risk_level = "WARNING"
        action = "Increase telemetry polling frequency and initiate pre-emptive RAID rebalancing."
    else:
        risk_level = "HEALTHY"
        action = "No intervention required. Normal operation."

    return PredictionResponse(
        serial_number=input_data.serial_number,
        failure_risk_score=round(prob, 4),
        predicted_failure_within_7d=is_failing,
        risk_level=risk_level,
        recommended_action=action,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_single(telemetry: DriveTelemetryInput):
    start_time = time.time()
    PREDICTION_REQUESTS.inc()
    try:
        bundle = get_model_bundle()
        resp = _score_drive(telemetry, bundle["model"], bundle["feature_cols"])
        PREDICTION_LATENCY.observe(time.time() - start_time)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    start_time = time.time()
    PREDICTION_REQUESTS.inc()
    try:
        bundle = get_model_bundle()
        model = bundle["model"]
        features = bundle["feature_cols"]

        results = [_score_drive(drive, model, features) for drive in batch.drives]
        critical = sum(1 for r in results if r.risk_level == "CRITICAL")
        warning = sum(1 for r in results if r.risk_level == "WARNING")

        PREDICTION_LATENCY.observe(time.time() - start_time)
        return BatchPredictionResponse(
            predictions=results,
            critical_count=critical,
            warning_count=warning,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
