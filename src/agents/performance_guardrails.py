"""Harness for the Performance Analyst: trend facts, prompt, validate, repair.

Pure logic stays LLM-free so CI can eval fixtures without Ollama.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import SystemMessage

from src.agents.llm import message_text

TRENDS = ("BULLISH", "BEARISH", "SIDEWAYS")
# Relative move below this → SIDEWAYS (matches persistence / noise).
DEFAULT_SIDEWAYS_THRESHOLD = 0.005
# Prefer decimals / $amounts so years like 2026 are not treated as prices.
_PRICE_RE = re.compile(
    r"\$([\d,]+\.\d+)|(?<![\d.$])(\d+\.\d{2,})(?![\d])"
)
_TREND_LINE_RE = re.compile(
    r"(?im)^\s*trend\s*:\s*(bullish|bearish|side[\s-]?ways|neutral)\b"
)


@dataclass(frozen=True)
class ForecastFacts:
    prices: tuple[float, ...]
    last_close: float | None
    expected_trend: str
    low: float | None
    high: float | None

    @property
    def allowed_prices(self) -> tuple[float, ...]:
        if self.last_close is None:
            return self.prices
        return (*self.prices, float(self.last_close))


@dataclass(frozen=True)
class GuardrailResult:
    ok: bool
    parsed_trend: str | None
    errors: tuple[str, ...]


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def extract_prices_from_forecast(forecast: dict[str, Any] | None) -> list[float]:
    """Prefer structured prediction points from get_forecast / predict_best."""
    if not forecast:
        return []
    prices: list[float] = []
    for point in forecast.get("predictions") or []:
        if not isinstance(point, dict):
            continue
        value = _as_float(point.get("value", point.get("close")))
        if value is not None:
            prices.append(value)
    return prices


def extract_prices_from_text(text: str) -> list[float]:
    """Parse dollar/decimal prices from forecast_text or model prose."""
    found: list[float] = []
    for match in _PRICE_RE.finditer(text or ""):
        raw = (match.group(1) or match.group(2) or "").replace(",", "")
        value = _as_float(raw)
        if value is not None:
            found.append(value)
    return found


def expected_trend_from_prices(
    prices: list[float],
    *,
    last_close: float | None = None,
    threshold: float = DEFAULT_SIDEWAYS_THRESHOLD,
) -> str:
    """Derive BULLISH / BEARISH / SIDEWAYS from the forecast path."""
    if not prices:
        return "SIDEWAYS"
    start = float(last_close) if last_close is not None else float(prices[0])
    end = float(prices[-1])
    if start == 0:
        return "SIDEWAYS"
    move = (end - start) / abs(start)
    if move >= threshold:
        return "BULLISH"
    if move <= -threshold:
        return "BEARISH"
    return "SIDEWAYS"


def build_forecast_facts(
    forecast: dict[str, Any] | None,
    forecast_text: str = "",
    *,
    threshold: float = DEFAULT_SIDEWAYS_THRESHOLD,
) -> ForecastFacts:
    prices = extract_prices_from_forecast(forecast)
    if not prices:
        # Skip the header line noise; keep numeric forecast lines.
        prices = extract_prices_from_text(forecast_text)
    last_close = _as_float((forecast or {}).get("last_close"))
    trend = expected_trend_from_prices(
        prices, last_close=last_close, threshold=threshold
    )
    low = min(prices) if prices else None
    high = max(prices) if prices else None
    return ForecastFacts(
        prices=tuple(prices),
        last_close=last_close,
        expected_trend=trend,
        low=low,
        high=high,
    )


def parse_trend_label(text: str) -> str | None:
    """Read structured `Trend:` first; otherwise keyword scan (sideways before bullish)."""
    if not text:
        return None
    line = _TREND_LINE_RE.search(text)
    if line:
        token = re.sub(r"[\s-]+", "", line.group(1).upper())
        if token in {"SIDEWAYS", "NEUTRAL"}:
            return "SIDEWAYS"
        if token in TRENDS:
            return token
    upper = text.upper()
    if "SIDEWAYS" in upper or "SIDE-WAYS" in upper or "SIDE WAYS" in upper:
        return "SIDEWAYS"
    if "BULLISH" in upper:
        return "BULLISH"
    if "BEARISH" in upper:
        return "BEARISH"
    return None


def _price_allowed(value: float, allowed: tuple[float, ...], *, atol: float) -> bool:
    return any(abs(value - ref) <= atol for ref in allowed)


def validate_performance_analysis(
    text: str,
    facts: ForecastFacts,
    *,
    atol: float | None = None,
) -> GuardrailResult:
    """Hard checks: trend matches numbers; no invented prices."""
    errors: list[str] = []
    parsed = parse_trend_label(text)
    if parsed is None:
        errors.append("missing_trend_label")
    elif parsed != facts.expected_trend:
        errors.append(
            f"trend_mismatch:got={parsed}:expected={facts.expected_trend}"
        )

    allowed = facts.allowed_prices
    if allowed:
        tolerance = atol if atol is not None else max(0.05, 0.001 * max(allowed))
        for price in extract_prices_from_text(text):
            if not _price_allowed(price, allowed, atol=tolerance):
                errors.append(f"invented_price:{price}")
                break
    return GuardrailResult(ok=not errors, parsed_trend=parsed, errors=tuple(errors))


def deterministic_performance_analysis(facts: ForecastFacts, ticker: str) -> str:
    """Code fallback when the LLM fails guardrails after retry."""
    if facts.low is not None and facts.high is not None:
        range_line = f"Range: {facts.low:.4f} – {facts.high:.4f} (from forecast only)"
    else:
        range_line = "Range: unavailable (no forecast prices)"
    return (
        f"Trend: {facts.expected_trend}\n"
        f"{range_line}\n"
        f"Caution: Forecast for {ticker} may be a weak or baseline path; "
        f"treat model uncertainty as high and do not invent levels."
    )


def repair_performance_analysis(
    text: str, facts: ForecastFacts, ticker: str
) -> str:
    """Force the correct Trend line; fall back to deterministic text if still invalid."""
    body = _TREND_LINE_RE.sub("", (text or "").strip()).strip()
    if not body:
        return deterministic_performance_analysis(facts, ticker)
    if facts.low is not None and facts.high is not None and not re.search(
        r"(?i)\brange\s*:", body
    ):
        body = f"Range: {facts.low:.4f} – {facts.high:.4f} (from forecast only)\n{body}"
    repaired = f"Trend: {facts.expected_trend}\n{body}".strip()
    if validate_performance_analysis(repaired, facts).ok:
        return repaired
    return deterministic_performance_analysis(facts, ticker)


def build_performance_prompt(ticker: str, forecast_text: str, facts: ForecastFacts) -> str:
    range_hint = (
        f"{facts.low:.4f} to {facts.high:.4f}"
        if facts.low is not None and facts.high is not None
        else "use only prices present in FORECAST DATA"
    )
    return f"""You are a Performance Analyst for equities.
Analyze this model forecast for {ticker}.

FORECAST DATA:
{forecast_text}

DETERMINISTIC FACTS (do not contradict):
- Required trend label: {facts.expected_trend}
- Allowed price range from forecast: {range_hint}

Write EXACTLY this structure (3 lines, then optional one short sentence):
Trend: {facts.expected_trend}
Range: <low> – <high> using only FORECAST DATA prices
Caution: <one sentence about model uncertainty>

Rules:
- The Trend line MUST be exactly: Trend: {facts.expected_trend}
- If prices are flat / nearly flat, that is SIDEWAYS — never call it Bullish or Bearish.
- Do not invent prices that are not in FORECAST DATA.
- Do not add Market Stance or Confidence lines.
"""


def run_performance_harness(
    *,
    ticker: str,
    forecast_text: str,
    forecast: dict[str, Any] | None,
    llm: Any,
    invoke: Callable[..., Any] | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Prompt → validate → retry → repair/fallback. Returns analysis + metadata."""
    facts = build_forecast_facts(forecast, forecast_text)
    call = invoke or (lambda messages: llm.invoke(messages))
    attempts: list[dict[str, Any]] = []
    text = ""
    result = GuardrailResult(ok=False, parsed_trend=None, errors=("not_run",))

    for attempt in range(1, max_attempts + 1):
        prompt = build_performance_prompt(ticker, forecast_text, facts)
        if attempt > 1:
            prompt += (
                "\n\nPREVIOUS OUTPUT FAILED GUARDRAILS. "
                f"Errors: {', '.join(result.errors)}. "
                f"Respond again with Trend: {facts.expected_trend} exactly."
            )
        response = call([SystemMessage(content=prompt)])
        text = message_text(response)
        result = validate_performance_analysis(text, facts)
        attempts.append(
            {
                "attempt": attempt,
                "ok": result.ok,
                "errors": list(result.errors),
                "parsed_trend": result.parsed_trend,
            }
        )
        if result.ok:
            break

    repaired = False
    if not result.ok:
        text = repair_performance_analysis(text, facts, ticker)
        result = validate_performance_analysis(text, facts)
        repaired = True
        if not result.ok:
            text = deterministic_performance_analysis(facts, ticker)
            result = validate_performance_analysis(text, facts)

    return {
        "performance_analysis": text,
        "performance_trend": facts.expected_trend,
        "performance_guardrail_ok": result.ok,
        "performance_guardrail_errors": list(result.errors),
        "performance_repaired": repaired,
        "performance_attempts": attempts,
        "forecast_facts": {
            "expected_trend": facts.expected_trend,
            "low": facts.low,
            "high": facts.high,
            "n_prices": len(facts.prices),
        },
    }
