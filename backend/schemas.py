from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=16)
    thread_id: str | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=16)
    horizon: int | None = Field(default=None, ge=1, le=60)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class TrainChildRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=16)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()


class PredictionPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int = Field(..., ge=1)
    value: float
    timestamp: str | None = None


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticker: str
    horizon: int
    predictions: list[PredictionPoint]
    model_version: str | None = None


class TaskAccepted(BaseModel):
    status: str = "queued"
    task_id: str


class TaskStatus(BaseModel):
    task_id: str
    task_type: str
    status: str
    result: Any | None = None
    error: str | None = None


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    ticker: str
    final_report: str | None = None
    recommendation: str | None = None
    confidence: str | None = None
    performance_analysis: str | None = None
    news_summary: str | None = None
    predictions: dict[str, Any] = Field(default_factory=dict)
    cached: bool = False
    detail: str | None = None
