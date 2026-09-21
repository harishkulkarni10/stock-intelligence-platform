"""Harness for the Risk Analyst (agent 4): facts, prompt, validate, repair.

Pure logic stays LLM-free so CI can eval fixtures without network or an LLM.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from logger.context import set_ticker
from logger.logger import get_logger, log_event
from src.agents.llm import message_text
from src.market.risk_metrics import RISK_LABELS, build_risk_metrics, format_risk_metrics_for_prompt

logger = get_logger("sip.agent4.harness")

MIN_ANALYSIS_CHARS = 160
_RISK_LINE_RE = re.compile(
    r"(?im)^\s*risk\s*:\s*(contained|moderate|elevated)\b"
)
_ANALYSIS_BLOCK_RE = re.compile(
    r"(?is)^\s*analysis\s*:\s*(.+?)(?=^\s*(?:market|financial|news|forecast|key downside|caveats?|risk)\s*:|\Z)",
    re.MULTILINE,
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.])(-?\d+(?:\.\d+)?)(?:\s*(?:%|pct|percent))?",
    re.IGNORECASE,
)
_ADVICE_RE = re.compile(
    r"(?i)\b(buy|sell|trim(?:\s+the)?\s+position|overweight|underweight|go long|go short|accumulate shares)\b"
)


@dataclass(frozen=True)
class RiskFacts:
    ticker: str
    allowed_risk: str
    model_source: str
    performance_trend: str | None
    performance_repaired: bool
    news_sentiment: str | None
    financial_health: str | None
    leverage_signal: str | None
    metrics: dict[str, float | int | None]
    drivers: tuple[str, ...]
    allowed_numbers: tuple[float, ...]
    status: str = "ok"


@dataclass(frozen=True)
class RiskGuardrailResult:
    ok: bool
    parsed_risk: str | None
    errors: tuple[str, ...]


def parse_risk_label(text: str) -> str | None:
    match = _RISK_LINE_RE.search(text or "")
    if not match:
        return None
    return match.group(1).upper()


def _analysis_body(text: str) -> str:
    match = _ANALYSIS_BLOCK_RE.search(text or "")
    if not match:
        return ""
    return match.group(1).strip()


def build_risk_facts(risk: dict[str, Any] | None, *, ticker: str = "") -> RiskFacts:
    payload = risk or {}
    allowed = str(payload.get("allowed_risk") or "MODERATE").upper()
    if allowed not in RISK_LABELS:
        allowed = "MODERATE"
    metrics_raw = payload.get("metrics") or {}
    metrics: dict[str, float | int | None] = {}
    for key, value in metrics_raw.items():
        if value is None:
            metrics[str(key)] = None
            continue
        try:
            if key == "history_points":
                metrics[str(key)] = int(value)
            else:
                metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            metrics[str(key)] = None
    allowed_nums = tuple(
        float(x)
        for x in (payload.get("allowed_numbers") or [])
        if isinstance(x, (int, float)) and not math.isnan(float(x))
    )
    drivers = tuple(str(x) for x in (payload.get("drivers") or []))
    return RiskFacts(
        ticker=str(payload.get("ticker") or ticker or "").upper(),
        allowed_risk=allowed,
        model_source=str(payload.get("model_source") or "unknown"),
        performance_trend=(
            str(payload.get("performance_trend")).upper()
            if payload.get("performance_trend")
            else None
        ),
        performance_repaired=bool(payload.get("performance_repaired")),
        news_sentiment=(
            str(payload.get("news_sentiment")).upper()
            if payload.get("news_sentiment")
            else None
        ),
        financial_health=(
            str(payload.get("financial_health")).upper()
            if payload.get("financial_health")
            else None
        ),
        leverage_signal=(
            str(payload.get("leverage_signal")).lower()
            if payload.get("leverage_signal")
            else None
        ),
        metrics=metrics,
        drivers=drivers,
        allowed_numbers=allowed_nums,
        status=str(payload.get("status") or "ok"),
    )


def _near_allowed(value: float, allowed: tuple[float, ...]) -> bool:
    if not allowed:
        return False
    for candidate in allowed:
        if candidate == 0 and abs(value) < 1e-6:
            return True
        tol = max(0.05 * abs(candidate), 0.15)
        if abs(value - candidate) <= tol:
            return True
        # Percent vs fraction (32 vs 0.32)
        if abs(value - candidate * 100.0) <= max(0.6, 0.08 * abs(candidate * 100.0)):
            return True
        if abs(value - candidate / 100.0) <= max(0.002, 0.08 * abs(candidate / 100.0)):
            return True
    return False


def validate_risk_analysis(text: str, facts: RiskFacts) -> RiskGuardrailResult:
    errors: list[str] = []
    parsed = parse_risk_label(text)
    if parsed is None:
        errors.append("missing_risk_label")
    elif parsed != facts.allowed_risk:
        errors.append(f"risk_mismatch:got={parsed}:expected={facts.allowed_risk}")

    body = _analysis_body(text)
    if len(body) < MIN_ANALYSIS_CHARS:
        errors.append("analysis_too_short")

    section_checks = (
        (r"(?im)^\s*market\b", "market"),
        (r"(?im)^\s*financial\b", "financial"),
        (r"(?im)^\s*news\b", "news"),
        (r"(?im)^\s*forecast\b", "forecast"),
        (r"(?im)^\s*key\s+downside", "key_downside"),
        (r"(?im)^\s*caveats?\s*:", "caveats"),
    )
    for pattern, name in section_checks:
        if not re.search(pattern, text or ""):
            errors.append(f"missing_section:{name}")

    if _ADVICE_RE.search(text or ""):
        errors.append("advice_language")

    # Ground numeric claims against allowed facts (skip years, day-counts, tiny ints).
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if abs(value) <= 60 and value == int(value):
            continue
        if 1900 <= value <= 2100 and value == int(value):
            continue
        if not _near_allowed(value, facts.allowed_numbers):
            errors.append(f"invented_number:{value}")
            break

    return RiskGuardrailResult(ok=not errors, parsed_risk=parsed, errors=tuple(errors))


def deterministic_risk_analysis(facts: RiskFacts) -> str:
    m = facts.metrics

    def pct_ann(key: str) -> str:
        value = m.get(key)
        return "n/a" if value is None else f"{float(value) * 100:.1f}%"

    def move() -> str:
        value = m.get("projected_move_pct")
        return "n/a" if value is None else f"{float(value):+.2f}%"

    def sigma() -> str:
        value = m.get("move_vs_daily_vol")
        return "n/a" if value is None else f"{float(value):.2f}"

    analysis = (
        f"Code-assigned risk for {facts.ticker} is {facts.allowed_risk} on this short "
        f"research window. Realized vol (~20d ann.) is {pct_ann('vol_ann_20d')}, "
        f"max drawdown in the history window is {pct_ann('max_drawdown')}, and the "
        f"projected path move is {move()} (~{sigma()} daily-vol units). "
        f"Context: trend={facts.performance_trend or 'n/a'}, "
        f"news={facts.news_sentiment or 'n/a'}, "
        f"financial health={facts.financial_health or 'n/a'}, "
        f"model source={facts.model_source}. "
        "This note restates those coded facts; it is research support only."
    )
    if len(analysis) < MIN_ANALYSIS_CHARS:
        analysis += (
            " Missing fields stay unknown rather than being filled with market folklore."
        )

    market = "- Path and realized volatility frame how jumpy the name has been lately."
    if "elevated_realized_volatility" in facts.drivers:
        market = "- Realized volatility screens elevated versus calm names on this window."
    financial = "- Fundamentals health/leverage enter only as coded signals, not new figures."
    if facts.financial_health == "STRESSED" or facts.leverage_signal == "stretched":
        financial = "- Financial health or leverage already screens stressed/stretched."
    news = "- News tone is treated as an event-risk hint, not a standalone thesis."
    if facts.news_sentiment == "NEGATIVE":
        news = "- Headline tone is negative, so event risk is flagged higher."
    elif facts.news_sentiment == "UNAVAILABLE":
        news = "- News coverage is thin or unavailable, so event risk is less visible."
    forecast = "- Forecast trust follows model source and whether the path note was repaired."
    if facts.model_source == "persistence" or facts.performance_repaired:
        forecast = (
            "- Forecast trust is weaker (persistence and/or repaired performance note)."
        )

    return (
        f"Risk: {facts.allowed_risk}\n"
        f"Analysis:\n{analysis}\n"
        f"Market / path risks:\n{market}\n"
        f"Financial risks:\n{financial}\n"
        f"News / event risks:\n{news}\n"
        f"Forecast / model risks:\n{forecast}\n"
        "Key downside scenarios:\n"
        "- Volatility expands and the short path overshoots typical daily moves.\n"
        "- News or balance-sheet stress arrives faster than the forecast window adapts.\n"
        "Caveats: Short-horizon research framing only — not portfolio VaR or trade advice."
    )


def repair_risk_analysis(text: str, facts: RiskFacts) -> str:
    body = _RISK_LINE_RE.sub("", (text or "").strip()).strip()
    if not body:
        return deterministic_risk_analysis(facts)
    if not re.search(r"(?im)^\s*analysis\s*:", body):
        body = f"Analysis:\n{body}"
    for label, header in (
        (r"(?im)^\s*market\b", "Market / path risks:\n- See coded volatility and path facts."),
        (r"(?im)^\s*financial\b", "Financial risks:\n- See coded health/leverage signals."),
        (r"(?im)^\s*news\b", "News / event risks:\n- See coded news sentiment."),
        (r"(?im)^\s*forecast\b", "Forecast / model risks:\n- See model source and repair flags."),
        (r"(?im)^\s*key\s+downside", "Key downside scenarios:\n- See volatility and event flags."),
        (r"(?im)^\s*caveats?\s*:", "Caveats: Research support only; not trade advice."),
    ):
        if not re.search(label, body):
            body = f"{body}\n{header}"
    repaired = f"Risk: {facts.allowed_risk}\n{body}".strip()
    if validate_risk_analysis(repaired, facts).ok:
        return repaired
    return deterministic_risk_analysis(facts)


def build_risk_prompt(
    ticker: str,
    risk_raw: str,
    facts: RiskFacts,
    *,
    performance_excerpt: str = "",
    news_excerpt: str = "",
    financial_excerpt: str = "",
) -> str:
    def clip(text: str, n: int = 700) -> str:
        t = (text or "").strip()
        return t if len(t) <= n else t[:n] + "…"

    return f"""You are a Risk Analyst for an equity research desk. Write a compact downside / uncertainty note for {ticker}.

RISK FACTS (code-computed; only source of numbers and the Risk label):
{risk_raw}

PRIOR AGENT NOTES (context only — do not invent new figures from them):
Performance:
{clip(performance_excerpt) or "(none)"}

Market:
{clip(news_excerpt) or "(none)"}

Financial:
{clip(financial_excerpt) or "(none)"}

LOCKED:
- Risk line MUST be exactly: Risk: {facts.allowed_risk}
- Use ONLY numbers present in RISK FACTS.
- Do not issue buy/sell or position advice.
- Do not override Performance trend with a new stance; explain fragility instead.

Use EXACTLY this structure:

Risk: {facts.allowed_risk}
Analysis:
<1–2 short paragraphs (~80–140 words). Explain why risk is {facts.allowed_risk} using the coded metrics and how they interact with trend/news/health.>
Market / path risks:
- <one grounded bullet>
Financial risks:
- <one grounded bullet>
News / event risks:
- <one grounded bullet>
Forecast / model risks:
- <one grounded bullet>
Key downside scenarios:
- <plausible short-horizon scenario grounded in the facts>
- <optional second scenario>
Caveats:
<1–2 sentences; research support only>

Rules:
- Medium length only — no essays.
- Vary wording; keep the same Risk label and facts.
- Missing metrics stay "unknown"; never invent vol, drawdown, or move %.
"""


def run_risk_harness(
    *,
    ticker: str,
    forecast: dict[str, Any] | None,
    performance_trend: str | None = None,
    performance_repaired: bool = False,
    performance_analysis: str | None = None,
    news_sentiment: str | None = None,
    news_summary: str | None = None,
    financial_health: str | None = None,
    financial_analysis: str | None = None,
    financials: dict[str, Any] | None = None,
    risk: dict[str, Any] | None = None,
    llm: Any = None,
    invoke: Callable[..., Any] | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Build facts → prompt → validate → retry → repair/fallback."""
    set_ticker(ticker)
    payload = risk or build_risk_metrics(
        ticker=ticker,
        forecast=forecast,
        performance_trend=performance_trend,
        performance_repaired=performance_repaired,
        news_sentiment=news_sentiment,
        financial_health=financial_health,
        financials=financials,
    )
    facts = build_risk_facts(payload, ticker=ticker)
    risk_raw = format_risk_metrics_for_prompt(payload)
    call = invoke or (lambda messages: llm.invoke(messages))
    attempts: list[dict[str, Any]] = []
    text = ""
    result = RiskGuardrailResult(ok=False, parsed_risk=None, errors=("not_run",))
    harness_started = time.perf_counter()

    log_event(
        logger,
        "agent4_harness_begin",
        step="agent4_begin",
        status="ok",
        data={"allowed_risk": facts.allowed_risk, "drivers": list(facts.drivers)},
    )

    if llm is None and invoke is None:
        text = deterministic_risk_analysis(facts)
        result = validate_risk_analysis(text, facts)
        return {
            "risk_analysis": text,
            "risk_level": facts.allowed_risk,
            "risk_guardrail_ok": result.ok,
            "risk_repaired": False,
            "risk": payload,
            "risk_attempts": [],
            "risk_duration_ms": round((time.perf_counter() - harness_started) * 1000, 2),
        }

    for attempt in range(1, max_attempts + 1):
        prompt = build_risk_prompt(
            ticker,
            risk_raw,
            facts,
            performance_excerpt=performance_analysis or "",
            news_excerpt=news_summary or "",
            financial_excerpt=financial_analysis or "",
        )
        if attempt > 1:
            prompt += (
                "\n\nPREVIOUS OUTPUT FAILED GUARDRAILS. "
                f"Errors: {', '.join(result.errors)}. "
                f"Respond again with Risk: {facts.allowed_risk} and only RISK FACTS numbers."
            )
            log_event(
                logger,
                "agent4_retry",
                step="agent4_retry",
                status="retry",
                metrics={"attempt": attempt},
                data={"prior_errors": list(result.errors)},
            )
        else:
            log_event(
                logger,
                "agent4_llm_invoke",
                step="agent4_llm",
                status="started",
                metrics={"attempt": attempt},
            )

        invoke_started = time.perf_counter()
        response = call(
            [
                SystemMessage(
                    content=(
                        "You write compact equity risk notes for a research desk. "
                        "Stay medium length, never invent metrics, never give trade advice."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        invoke_ms = (time.perf_counter() - invoke_started) * 1000
        text = message_text(response)
        result = validate_risk_analysis(text, facts)
        attempts.append(
            {
                "attempt": attempt,
                "ok": result.ok,
                "errors": list(result.errors),
                "duration_ms": round(invoke_ms, 2),
            }
        )
        log_event(
            logger,
            "agent4_validation",
            step="agent4_validation",
            status="success" if result.ok else "failed",
            duration_ms=invoke_ms,
            metrics={"attempt": attempt, "chars": len(text or ""), "ok": result.ok},
            data={"risk": result.parsed_risk, "errors": list(result.errors)},
        )
        if result.ok:
            break

    repaired = False
    if not result.ok:
        log_event(
            logger,
            "agent4_repair",
            level=logging.WARNING,
            step="agent4_repair",
            status="repair",
            metrics={"after_attempts": max_attempts},
        )
        text = repair_risk_analysis(text, facts)
        result = validate_risk_analysis(text, facts)
        repaired = True
        if not result.ok:
            log_event(
                logger,
                "agent4_deterministic_fallback",
                level=logging.WARNING,
                step="agent4_fallback",
                status="fallback",
            )
            text = deterministic_risk_analysis(facts)
            result = validate_risk_analysis(text, facts)

    total_ms = (time.perf_counter() - harness_started) * 1000
    log_event(
        logger,
        "agent4_harness_end",
        step="agent4_end",
        status="success" if result.ok else "failed",
        duration_ms=total_ms,
        metrics={"repaired": repaired, "chars": len(text or ""), "attempts": len(attempts)},
        data={"risk": facts.allowed_risk, "guardrail_ok": result.ok},
    )

    return {
        "risk_analysis": text,
        "risk_level": facts.allowed_risk,
        "risk_guardrail_ok": result.ok,
        "risk_guardrail_errors": list(result.errors),
        "risk_repaired": repaired,
        "risk_attempts": attempts,
        "risk_duration_ms": round(total_ms, 2),
        "risk": payload,
    }
