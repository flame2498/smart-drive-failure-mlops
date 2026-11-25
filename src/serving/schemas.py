"""Pydantic schemas for request validation in FastAPI serving."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DriveTelemetryInput(BaseModel):
    serial_number: str = Field(..., example="WDC-000123")
    model: str = Field(..., example="ST12000NM0007")
    smart_5_raw: float = Field(0.0, description="Reallocated Sectors Count", ge=0.0)
    smart_9_raw: float = Field(..., description="Power-On Hours", ge=0.0)
    smart_187_raw: float = Field(0.0, description="Reported Uncorrectable Errors", ge=0.0)
    smart_188_raw: float = Field(0.0, description="Command Timeout", ge=0.0)
    smart_194_raw: float = Field(32.0, description="Temperature (°C)", ge=0.0, le=100.0)
    smart_197_raw: float = Field(0.0, description="Current Pending Sector Count", ge=0.0)
    smart_5_raw_mean_7d: Optional[float] = Field(0.0, ge=0.0)
    smart_5_raw_delta_7d: Optional[float] = Field(0.0)
    smart_187_raw_mean_7d: Optional[float] = Field(0.0, ge=0.0)
    smart_187_raw_delta_7d: Optional[float] = Field(0.0)
    smart_197_raw_mean_7d: Optional[float] = Field(0.0, ge=0.0)
    smart_197_raw_delta_7d: Optional[float] = Field(0.0)
    smart_194_raw_mean_7d: Optional[float] = Field(32.0, ge=0.0)
    smart_194_raw_delta_7d: Optional[float] = Field(0.0)


class PredictionResponse(BaseModel):
    serial_number: str
    failure_risk_score: float = Field(..., ge=0.0, le=1.0)
    predicted_failure_within_7d: int = Field(..., description="1 if failure is imminent within 7 days, 0 otherwise")
    risk_level: str = Field(..., description="'CRITICAL', 'WARNING', or 'HEALTHY'")
    recommended_action: str


class BatchPredictionRequest(BaseModel):
    drives: List[DriveTelemetryInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    critical_count: int
    warning_count: int
