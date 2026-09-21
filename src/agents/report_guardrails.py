"""Harness for the Report / Synthesis agent: facts, prompt, validate, repair.

Composes specialist notes into one research brief. Numbers stay tool-grounded;
stance and confidence are locked from prior agents / forecast trust code.
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
from src.agents.performance_guardrails import projected_move_pct
from src.market.risk_metrics import RISK_LABELS

logger = get_logger("sip.agent5.harness")

STANCES = ("BULLISH", "BEARISH", "NEUTRAL")
CONFIDENCE_LABELS = ("High", "Medium", "Low")
MIN_EXEC_CHARS = 120
MIN_TOTAL_CHARS = 420

_STANCE_LINE_RE = re.compile(
    r"(?im)^\s*stance\s*:\s*(bullish|bearish|neutral)\b"
)
_CONF_LINE_RE = re.compile(
    r"(?im)^\s*confidence\s*:\s*(high|medium|low)\b"
)
_EXEC_BLOCK_RE = re.compile(
    r"(?is)^\s*executive\s+summary\s*:\s*(.+?)(?=^\s*(?:forecast|news|fundamentals|risk|bull|bear|key|caveats?|stance|confidence)\s*:|\Z)",
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
class ReportFacts:
    ticker: str
    stance: str
    confidence: str
    performance_trend: str
    news_sentiment: str
    financial_health: str
    risk_level: str
    projected_move_pct: float | None
    model_source: str
    allowed_numbers: tuple[float, ...]
    performance_excerpt: str = ""
    news_excerpt: str = ""
    financial_excerpt: str = ""
    risk_excerpt: str = ""
    forecast_excerpt: str = ""


@dataclass(frozen=True)
class ReportGuardrailResult:
    ok: bool
    parsed_stance: str | None
    parsed_confidence: str | None
    errors: tuple[str, ...]


def stance_from_trend(trend: str | None) -> str:
    label = (trend or "SIDEWAYS").upper()
    if label == "SIDEWAYS":
        return "NEUTRAL"
    if label in {"BULLISH", "BEARISH"}:
        return label
    return "NEUTRAL"


def normalize_confidence(value: str | None) -> str:
    raw = (value or "Medium").strip().capitalize()
    if raw.lower() == "high":
        return "High"
    if raw.lower() == "low":
        return "Low"
    return "Medium"


def parse_stance_label(text: str) -> str | None:
    match = _STANCE_LINE_RE.search(text or "")
    if not match:
        return None
    return match.group(1).upper()


def parse_confidence_label(text: str) -> str | None:
    match = _CONF_LINE_RE.search(text or "")
    if not match:
        return None
    return match.group(1).capitalize()


def _clip(text: str, n: int = 900) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= n:
        return cleaned
    return cleaned[:n] + "…"


def _collect_allowed_numbers(
    *,
    forecast: dict[str, Any] | None,
    risk: dict[str, Any] | None,
    financials: dict[str, Any] | None,
    move_pct: float | None,
) -> tuple[float, ...]:
    allowed: list[float] = []
    data = forecast or {}
    for key in ("last_close",):
        try:
            value = float(data.get(key))
            if math.isfinite(value):
                allowed.append(value)
        except (TypeError, ValueError):
            pass
    for point in data.get("predictions") or []:
        if not isinstance(point, dict):
            continue
        try:
            value = float(point.get("value"))
            if math.isfinite(value):
                allowed.append(value)
        except (TypeError, ValueError):
            pass
    if move_pct is not None and math.isfinite(move_pct):
        allowed.append(float(move_pct))
        allowed.append(abs(float(move_pct)))

    risk_payload = risk or {}
    for value in risk_payload.get("allowed_numbers") or []:
        try:
            num = float(value)
            if math.isfinite(num):
                allowed.append(num)
        except (TypeError, ValueError):
            pass
    metrics = risk_payload.get("metrics") if isinstance(risk_payload.get("metrics"), dict) else {}
    for value in metrics.values():
        try:
            num = float(value)
            if math.isfinite(num):
                allowed.append(num)
                if abs(num) <= 2:
                    allowed.append(round(abs(num) * 100.0, 2))
        except (TypeError, ValueError):
            pass

    fin = financials or {}
    for value in fin.get("allowed_numbers") or []:
        try:
            num = float(value)
            if math.isfinite(num):
                allowed.append(num)
        except (TypeError, ValueError):
            pass

    # Deduplicate with light rounding.
    uniq: list[float] = []
    seen: set[float] = set()
    for num in allowed:
        key = round(num, 6)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(num))
    return tuple(uniq)


def build_report_facts(
    *,
    ticker: str,
    forecast: dict[str, Any] | None = None,
    forecast_text: str = "",
    performance_trend: str | None = None,
    performance_analysis: str = "",
    news_sentiment: str | None = None,
    news_summary: str = "",
    financial_health: str | None = None,
    financial_analysis: str = "",
    financials: dict[str, Any] | None = None,
    risk_level: str | None = None,
    risk_analysis: str = "",
    risk: dict[str, Any] | None = None,
    confidence: str | None = None,
) -> ReportFacts:
    trend = (performance_trend or "SIDEWAYS").upper()
    if trend not in {"BULLISH", "BEARISH", "SIDEWAYS"}:
        trend = "SIDEWAYS"
    stance = stance_from_trend(trend)
    sentiment = (news_sentiment or "MIXED").upper() or "MIXED"
    health = (financial_health or "UNAVAILABLE").upper() or "UNAVAILABLE"
    risk_label = (risk_level or (risk or {}).get("allowed_risk") or "MODERATE").upper()
    if risk_label not in RISK_LABELS:
        risk_label = "MODERATE"
    move = projected_move_pct(forecast)
    return ReportFacts(
        ticker=str(ticker or "").upper(),
        stance=stance,
        confidence=normalize_confidence(confidence),
        performance_trend=trend,
        news_sentiment=sentiment,
        financial_health=health,
        risk_level=risk_label,
        projected_move_pct=move,
        model_source=str((forecast or {}).get("model_source") or "unknown"),
        allowed_numbers=_collect_allowed_numbers(
            forecast=forecast,
            risk=risk,
            financials=financials,
            move_pct=move,
        ),
        performance_excerpt=_clip(performance_analysis),
        news_excerpt=_clip(news_summary),
        financial_excerpt=_clip(financial_analysis),
        risk_excerpt=_clip(risk_analysis),
        forecast_excerpt=_clip(forecast_text, 1200),
    )


def _near_allowed(value: float, allowed: tuple[float, ...]) -> bool:
    if not allowed:
        return False
    for candidate in allowed:
        if candidate == 0 and abs(value) < 1e-6:
            return True
        tol = max(0.05 * abs(candidate), 0.25)
        if abs(value - candidate) <= tol:
            return True
        if abs(value - candidate * 100.0) <= max(0.8, 0.08 * abs(candidate * 100.0)):
            return True
        if abs(value - candidate / 100.0) <= max(0.002, 0.08 * abs(candidate / 100.0)):
            return True
    return False


def validate_report(text: str, facts: ReportFacts) -> ReportGuardrailResult:
    errors: list[str] = []
    parsed_stance = parse_stance_label(text)
    parsed_conf = parse_confidence_label(text)

    if parsed_stance is None:
        errors.append("missing_stance_label")
    elif parsed_stance != facts.stance:
        errors.append(f"stance_mismatch:got={parsed_stance}:expected={facts.stance}")

    if parsed_conf is None:
        errors.append("missing_confidence_label")
    elif parsed_conf != facts.confidence:
        errors.append(
            f"confidence_mismatch:got={parsed_conf}:expected={facts.confidence}"
        )

    exec_body = ""
    match = _EXEC_BLOCK_RE.search(text or "")
    if match:
        exec_body = match.group(1).strip()
    if len(exec_body) < MIN_EXEC_CHARS:
        errors.append("executive_summary_too_short")
    if len((text or "").strip()) < MIN_TOTAL_CHARS:
        errors.append("report_too_short")

    section_checks = (
        (r"(?im)^\s*executive\s+summary\s*:", "executive_summary"),
        (r"(?im)^\s*forecast\b", "forecast"),
        (r"(?im)^\s*news\b", "news"),
        (r"(?im)^\s*fundamentals\b", "fundamentals"),
        (r"(?im)^\s*risk\b", "risk"),
        (r"(?im)^\s*bull\s+case\s*:", "bull_case"),
        (r"(?im)^\s*bear\s+case\s*:", "bear_case"),
        (r"(?im)^\s*key\s+drivers\s*:", "key_drivers"),
        (r"(?im)^\s*key\s+risks\s*:", "key_risks"),
        (r"(?im)^\s*caveats?\s*:", "caveats"),
    )
    for pattern, name in section_checks:
        if not re.search(pattern, text or ""):
            errors.append(f"missing_section:{name}")

    # Locked specialist labels must appear so the brief cannot drift.
    upper = (text or "").upper()
    for label, name in (
        (facts.performance_trend, "trend_label"),
        (facts.news_sentiment, "news_label"),
        (facts.financial_health, "health_label"),
        (facts.risk_level, "risk_label"),
    ):
        if label and label.upper() not in upper:
            errors.append(f"missing_locked_label:{name}")

    if _ADVICE_RE.search(text or ""):
        errors.append("advice_language")

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

    return ReportGuardrailResult(
        ok=not errors,
        parsed_stance=parsed_stance,
        parsed_confidence=parsed_conf,
        errors=tuple(errors),
    )


def _move_str(facts: ReportFacts) -> str:
    if facts.projected_move_pct is None:
        return "n/a"
    return f"{facts.projected_move_pct:+.2f}%"


def deterministic_report(facts: ReportFacts) -> str:
    move = _move_str(facts)
    exec_summary = (
        f"{facts.ticker} research brief on this short horizon: stance {facts.stance} "
        f"(from Performance trend {facts.performance_trend}), confidence {facts.confidence}, "
        f"news {facts.news_sentiment}, fundamentals {facts.financial_health}, "
        f"risk {facts.risk_level}, projected path move {move}. "
        "This memo only restates the specialist packet and forecast facts — "
        "it does not invent prices, headlines, or statement figures."
    )
    if len(exec_summary) < MIN_EXEC_CHARS:
        exec_summary += (
            " Read the specialist tabs for detail; disagreement across labels is useful signal."
        )

    return (
        f"Stance: {facts.stance}\n"
        f"Confidence: {facts.confidence}\n\n"
        f"Executive summary:\n{exec_summary}\n\n"
        f"Forecast / performance:\n"
        f"Performance labeled the path {facts.performance_trend}. "
        f"Projected move from last close to the final forecast session is {move}. "
        "Treat the chart path as the only price source for this brief.\n\n"
        f"News:\n"
        f"Market expert tone is {facts.news_sentiment}. "
        "Headline context can change quickly and does not override the path numbers.\n\n"
        f"Fundamentals:\n"
        f"Financial health is {facts.financial_health} on the coded snapshot. "
        "Missing coverage stays unknown rather than invented.\n\n"
        f"Risk:\n"
        f"Risk analyst label is {facts.risk_level}, summarizing downside and uncertainty "
        "from path volatility and the other coded context flags.\n\n"
        "Bull case:\n"
        f"- Path/news/financial labels that support constructive reading stay in force "
        f"(trend {facts.performance_trend}, news {facts.news_sentiment}).\n\n"
        "Bear case:\n"
        f"- Risk {facts.risk_level} and any stressed or negative labels caution against "
        "treating a calm narrative as certainty.\n\n"
        "Key drivers:\n"
        f"- Forecast path move {move} and Performance trend {facts.performance_trend}.\n"
        f"- News tone {facts.news_sentiment} and financial health {facts.financial_health}.\n\n"
        "Key risks:\n"
        f"- Coded risk level {facts.risk_level} on this short research window.\n"
        "- Specialist disagreement or thin coverage can weaken conviction.\n\n"
        "Caveats: Research support only — not trade advice. Numbers come from the "
        "forecast and specialist facts objects, not from this synthesizer."
    )


def repair_report(text: str, facts: ReportFacts) -> str:
    body = (text or "").strip()
    body = _STANCE_LINE_RE.sub("", body).strip()
    body = _CONF_LINE_RE.sub("", body).strip()
    if not body or len(body) < MIN_EXEC_CHARS:
        return deterministic_report(facts)

    if not re.search(r"(?im)^\s*executive\s+summary\s*:", body):
        body = f"Executive summary:\n{body}"
    for pattern, header in (
        (r"(?im)^\s*forecast\b", "Forecast / performance:\n- See Performance note and chart path."),
        (r"(?im)^\s*news\b", f"News:\n- Tone {facts.news_sentiment} from Market expert."),
        (r"(?im)^\s*fundamentals\b", f"Fundamentals:\n- Health {facts.financial_health}."),
        (r"(?im)^\s*risk\b", f"Risk:\n- Label {facts.risk_level}."),
        (r"(?im)^\s*bull\s+case\s*:", "Bull case:\n- Constructive labels in the packet, if any."),
        (r"(?im)^\s*bear\s+case\s*:", "Bear case:\n- Caution from risk or adverse labels."),
        (r"(?im)^\s*key\s+drivers\s*:", "Key drivers:\n- Path, news, and fundamentals labels."),
        (r"(?im)^\s*key\s+risks\s*:", f"Key risks:\n- Risk {facts.risk_level}."),
        (r"(?im)^\s*caveats?\s*:", "Caveats: Research support only."),
    ):
        if not re.search(pattern, body):
            body = f"{body}\n\n{header}"

    # Ensure locked labels appear somewhere.
    for label in (
        facts.performance_trend,
        facts.news_sentiment,
        facts.financial_health,
        facts.risk_level,
    ):
        if label.upper() not in body.upper():
            body = f"{body}\nLocked labels: {facts.performance_trend}, {facts.news_sentiment}, {facts.financial_health}, {facts.risk_level}."
            break

    repaired = (
        f"Stance: {facts.stance}\n"
        f"Confidence: {facts.confidence}\n\n"
        f"{body}"
    ).strip()
    if validate_report(repaired, facts).ok:
        return repaired
    return deterministic_report(facts)


def build_report_prompt(facts: ReportFacts) -> str:
    move = _move_str(facts)
    return f"""You are the Report / Synthesis agent for an equity research desk.
Write one compact research brief for {facts.ticker} that stitches the specialist packet.

LOCKED LABELS (must keep exactly):
- Stance: {facts.stance}  (from Performance trend {facts.performance_trend})
- Confidence: {facts.confidence}
- News tone: {facts.news_sentiment}
- Financial health: {facts.financial_health}
- Risk: {facts.risk_level}
- Projected move: {move}
- Model source (context only): {facts.model_source}

FORECAST DATA (only source of prices):
{facts.forecast_excerpt or "(none)"}

PERFORMANCE NOTE:
{facts.performance_excerpt or "(none)"}

NEWS NOTE:
{facts.news_excerpt or "(none)"}

FINANCIAL NOTE:
{facts.financial_excerpt or "(none)"}

RISK NOTE:
{facts.risk_excerpt or "(none)"}

Use EXACTLY this structure:

Stance: {facts.stance}
Confidence: {facts.confidence}

Executive summary:
<2–4 sentences (~120–180 chars minimum of substance). Overall picture using the locked labels. No trade advice.>

Forecast / performance:
<short paragraph grounded in path + projected move {move}; trend is {facts.performance_trend}.>

News:
<short paragraph; tone is {facts.news_sentiment}. Do not invent headlines.>

Fundamentals:
<short paragraph; health is {facts.financial_health}. Do not invent statement figures.>

Risk:
<short paragraph; risk is {facts.risk_level}.>

Bull case:
- <one grounded bullet>
- <optional second bullet>

Bear case:
- <one grounded bullet>
- <optional second bullet>

Key drivers:
- <grounded bullet>
- <grounded bullet>

Key risks:
- <grounded bullet>
- <grounded bullet>

Caveats:
<1–2 sentences; research support only>

Rules:
- Medium length — useful brief, not an essay (roughly 350–500 words max).
- Stance and Confidence lines MUST match the locked values.
- Every locked specialist label ({facts.performance_trend}, {facts.news_sentiment}, {facts.financial_health}, {facts.risk_level}) MUST appear in the body.
- Use ONLY numbers present in FORECAST DATA / specialist facts (projected move {move} is allowed).
- Do not invent prices, headlines, or fundamentals.
- No buy/sell or position advice.
- Do not override specialist labels with a new thesis.
"""


def run_report_harness(
    *,
    ticker: str,
    forecast: dict[str, Any] | None = None,
    forecast_text: str = "",
    performance_trend: str | None = None,
    performance_analysis: str = "",
    news_sentiment: str | None = None,
    news_summary: str = "",
    financial_health: str | None = None,
    financial_analysis: str = "",
    financials: dict[str, Any] | None = None,
    risk_level: str | None = None,
    risk_analysis: str = "",
    risk: dict[str, Any] | None = None,
    confidence: str | None = None,
    llm: Any = None,
    invoke: Callable[..., Any] | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Prompt → validate → retry → repair/fallback for the synthesis brief."""
    set_ticker(ticker)
    facts = build_report_facts(
        ticker=ticker,
        forecast=forecast,
        forecast_text=forecast_text,
        performance_trend=performance_trend,
        performance_analysis=performance_analysis,
        news_sentiment=news_sentiment,
        news_summary=news_summary,
        financial_health=financial_health,
        financial_analysis=financial_analysis,
        financials=financials,
        risk_level=risk_level,
        risk_analysis=risk_analysis,
        risk=risk,
        confidence=confidence,
    )
    call = invoke or (lambda messages: llm.invoke(messages))
    attempts: list[dict[str, Any]] = []
    text = ""
    result = ReportGuardrailResult(
        ok=False, parsed_stance=None, parsed_confidence=None, errors=("not_run",)
    )
    harness_started = time.perf_counter()

    log_event(
        logger,
        "agent5_harness_begin",
        step="agent5_begin",
        status="ok",
        data={
            "stance": facts.stance,
            "confidence": facts.confidence,
            "risk": facts.risk_level,
            "trend": facts.performance_trend,
        },
    )

    if llm is None and invoke is None:
        text = deterministic_report(facts)
        result = validate_report(text, facts)
        return {
            "final_report": text,
            "draft_report": text,
            "recommendation": facts.stance,
            "confidence": facts.confidence,
            "report_guardrail_ok": result.ok,
            "report_repaired": False,
            "report_attempts": [],
            "report_duration_ms": round((time.perf_counter() - harness_started) * 1000, 2),
        }

    for attempt in range(1, max_attempts + 1):
        prompt = build_report_prompt(facts)
        if attempt > 1:
            prompt += (
                "\n\nPREVIOUS OUTPUT FAILED GUARDRAILS. "
                f"Errors: {', '.join(result.errors)}. "
                f"Respond again with Stance: {facts.stance}, Confidence: {facts.confidence}, "
                "all required sections, and only allowed numbers."
            )
            log_event(
                logger,
                "agent5_retry",
                step="agent5_retry",
                status="retry",
                metrics={"attempt": attempt},
                data={"prior_errors": list(result.errors)},
            )
        else:
            log_event(
                logger,
                "agent5_llm_invoke",
                step="agent5_llm",
                status="started",
                metrics={"attempt": attempt},
            )

        invoke_started = time.perf_counter()
        response = call(
            [
                SystemMessage(
                    content=(
                        "You write compact equity research briefs that synthesize "
                        "specialist notes. Never invent numbers or give trade advice."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        invoke_ms = (time.perf_counter() - invoke_started) * 1000
        text = message_text(response)
        result = validate_report(text, facts)
        attempts.append(
            {
                "attempt": attempt,
                "ok": result.ok,
                "errors": list(result.errors),
                "parsed_stance": result.parsed_stance,
                "parsed_confidence": result.parsed_confidence,
                "chars": len(text or ""),
                "duration_ms": round(invoke_ms, 2),
            }
        )
        log_event(
            logger,
            "agent5_validation",
            step="agent5_validation",
            status="success" if result.ok else "failed",
            duration_ms=invoke_ms,
            metrics={"attempt": attempt, "chars": len(text or ""), "ok": result.ok},
            data={
                "stance": result.parsed_stance,
                "confidence": result.parsed_confidence,
                "errors": list(result.errors),
            },
        )
        if result.ok:
            break

    repaired = False
    if not result.ok:
        log_event(
            logger,
            "agent5_repair",
            level=logging.WARNING,
            step="agent5_repair",
            status="repair",
            metrics={"after_attempts": max_attempts},
        )
        text = repair_report(text, facts)
        result = validate_report(text, facts)
        repaired = True
        if not result.ok:
            log_event(
                logger,
                "agent5_deterministic_fallback",
                level=logging.WARNING,
                step="agent5_fallback",
                status="fallback",
            )
            text = deterministic_report(facts)
            result = validate_report(text, facts)

    total_ms = (time.perf_counter() - harness_started) * 1000
    log_event(
        logger,
        "agent5_harness_end",
        step="agent5_end",
        status="success" if result.ok else "failed",
        duration_ms=total_ms,
        metrics={
            "repaired": repaired,
            "chars": len(text or ""),
            "attempts": len(attempts),
        },
        data={"stance": facts.stance, "guardrail_ok": result.ok},
    )

    return {
        "final_report": text,
        "draft_report": text,
        "recommendation": facts.stance,
        "confidence": facts.confidence,
        "report_guardrail_ok": result.ok,
        "report_guardrail_errors": list(result.errors),
        "report_repaired": repaired,
        "report_attempts": attempts,
        "report_duration_ms": round(total_ms, 2),
    }
