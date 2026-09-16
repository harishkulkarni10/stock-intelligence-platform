from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=16)
    thread_id: str | None = None
    force_refresh: bool = False

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


class HelpChatMessage(BaseModel):
    role: str = Field(..., min_length=1, max_length=16)
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("role")
    @classmethod
    def normalize_role(cls, value: str) -> str:
        role = value.strip().lower()
        if role not in {"user", "assistant"}:
            raise ValueError("role must be user or assistant")
        return role


class HelpChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[HelpChatMessage] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("message cannot be empty")
        return text


class HelpChatResponse(BaseModel):
    reply: str
    out_of_scope: bool = False


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    ticker: str
    mode: str | None = "performance_news_financial"
    final_report: str | None = None
    recommendation: str | None = None
    confidence: str | None = None
    performance_analysis: str | None = None
    performance_trend: str | None = None
    performance_guardrail_ok: bool | None = None
    performance_repaired: bool | None = None
    news_summary: str | None = None
    news_sentiment: str | None = None
    news_guardrail_ok: bool | None = None
    news_repaired: bool | None = None
    news: dict[str, Any] = Field(default_factory=dict)
    financial_analysis: str | None = None
    financial_health: str | None = None
    financial_guardrail_ok: bool | None = None
    financial_repaired: bool | None = None
    financials: dict[str, Any] = Field(default_factory=dict)
    company: dict[str, Any] = Field(default_factory=dict)
    draft_report: str | None = None
    predictions: dict[str, Any] = Field(default_factory=dict)
    metric_explanations: dict[str, str] = Field(default_factory=dict)
    cached: bool = False
    cached_at_ts: int | None = None
    cache_age_seconds: int | None = None
    cache_ttl_seconds: int | None = None
    detail: str | None = None
    trace_id: str | None = None
