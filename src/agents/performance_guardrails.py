"""Harness for the Performance Analyst: trend facts, prompt, validate, repair.

Pure logic stays LLM-free so CI can eval fixtures without Ollama.
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

logger = get_logger("sip.agent1.harness")

TRENDS = ("BULLISH", "BEARISH", "SIDEWAYS")
DEFAULT_SIDEWAYS_THRESHOLD = 0.005
MIN_ANALYSIS_CHARS = 180
# Prefer $amounts / multi-digit decimals; skip years and percentages like 0.72%.
_PRICE_RE = re.compile(
    r"\$([\d,]+\.\d+)|(?<![\d.$])(\d{2,}\.\d{2,})(?!\d)(?!\s*(?:%|percent|pct)\b)"
)
_TREND_LINE_RE = re.compile(
    r"(?im)^\s*trend\s*:\s*(bullish|bearish|side[\s-]?ways|neutral)\b"
)
_ANALYSIS_BLOCK_RE = re.compile(
    r"(?is)^\s*analysis\s*:\s*(.+?)(?=^\s*(?:key points|caveats?|range|trend)\s*:|\Z)",
    re.MULTILINE,
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


def _analysis_body(text: str) -> str:
    match = _ANALYSIS_BLOCK_RE.search(text or "")
    if match:
        return match.group(1).strip()
    # Fallback: everything after the first Analysis: label on one line.
    for line in (text or "").splitlines():
        if re.match(r"(?i)^\s*analysis\s*:", line):
            return re.sub(r"(?i)^\s*analysis\s*:", "", line).strip()
    return ""


def validate_performance_analysis(
    text: str,
    facts: ForecastFacts,
    *,
    atol: float | None = None,
) -> GuardrailResult:
    """Hard checks: trend matches numbers; no invented prices; real Analysis section."""
    errors: list[str] = []
    parsed = parse_trend_label(text)
    if parsed is None:
        errors.append("missing_trend_label")
    elif parsed != facts.expected_trend:
        errors.append(
            f"trend_mismatch:got={parsed}:expected={facts.expected_trend}"
        )

    if not re.search(r"(?im)^\s*analysis\s*:", text or ""):
        errors.append("missing_analysis_section")
    else:
        body = _analysis_body(text)
        if len(body) < MIN_ANALYSIS_CHARS:
            errors.append(f"analysis_too_short:{len(body)}<{MIN_ANALYSIS_CHARS}")

    allowed = facts.allowed_prices
    if allowed:
        tolerance = atol if atol is not None else max(0.05, 0.001 * max(allowed))
        for price in extract_prices_from_text(text):
            if not _price_allowed(price, allowed, atol=tolerance):
                errors.append(f"invented_price:{price}")
                break
    return GuardrailResult(ok=not errors, parsed_trend=parsed, errors=tuple(errors))


def _path_narrative(facts: ForecastFacts, ticker: str) -> str:
    prices = list(facts.prices)
    if not prices:
        return (
            f"No usable forecast sessions were available for {ticker}, so the path "
            "cannot be interpreted beyond a placeholder SIDEWAYS framing."
        )
    last = facts.last_close
    start = float(last) if last is not None else float(prices[0])
    end = float(prices[-1])
    move_pct = ((end - start) / abs(start) * 100.0) if start else 0.0
    move_words = f"{move_pct:+.2f} percent"
    return (
        f"For {ticker}, the path ends at {end:.4f} versus last close {start:.4f} "
        f"({move_words}), spanning {facts.low:.4f} to {facts.high:.4f}. "
        f"That supports a {facts.expected_trend} label on this short horizon: "
        "a clear staircase looks more decisive than a narrow or choppy band. "
        "Treat it as chart framing only — a few model sessions, not a valuation call."
    )


def deterministic_performance_analysis(facts: ForecastFacts, ticker: str) -> str:
    """Code fallback when the LLM fails guardrails after retry."""
    if facts.low is not None and facts.high is not None:
        range_line = f"Range: {facts.low:.4f} – {facts.high:.4f}"
    else:
        range_line = "Range: unavailable"
    narrative = _path_narrative(facts, ticker)
    return (
        f"Trend: {facts.expected_trend}\n"
        f"{range_line}\n"
        f"Analysis:\n{narrative}\n"
        "Key points:\n"
        f"- Label is {facts.expected_trend} from last close to the final forecast session.\n"
        "- Range is only the min/max of those forecasted prices.\n"
        "- Short horizons and flat paths mean lower directional conviction.\n"
        "Caveats: Model paths can miss news shocks; research support only, not advice."
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
        body = f"Range: {facts.low:.4f} – {facts.high:.4f}\n{body}"
    if not re.search(r"(?im)^\s*analysis\s*:", body):
        body = f"{body}\nAnalysis:\n{_path_narrative(facts, ticker)}"
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
    session_lines = "\n".join(
        f"  - session {i + 1}: {price:.4f}" for i, price in enumerate(facts.prices)
    ) or "  - (no sessions)"
    return f"""You are a Performance Analyst. Write a compact research note for {ticker} — useful and specific, not an essay.

FORECAST DATA:
{forecast_text}

SESSIONS:
{session_lines}
Last close: {facts.last_close}

LOCKED:
- Trend line MUST be: Trend: {facts.expected_trend}
- Range only from forecast prices: {range_hint}

Use EXACTLY this structure:

Trend: {facts.expected_trend}
Range: <low> – <high>
Analysis:
<1–2 short paragraphs (about 80–140 words total). Cover path shape, move vs last close, and how decisive {facts.expected_trend} looks. No filler.>
Key points:
- <one crisp sentence>
- <one crisp sentence>
- <one crisp sentence>
Caveats:
<1–2 sentences on uncertainty / not advice>

Rules:
- Medium length only — do not write long essays or step-by-step % for every session.
- Vary wording from run to run; keep the same facts and Trend label.
- Do not invent prices outside FORECAST DATA.
- Do not discuss news.
- Flat / near-flat paths → SIDEWAYS.
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
    set_ticker(ticker)
    facts = build_forecast_facts(forecast, forecast_text)
    call = invoke or (lambda messages: llm.invoke(messages))
    attempts: list[dict[str, Any]] = []
    text = ""
    result = GuardrailResult(ok=False, parsed_trend=None, errors=("not_run",))
    harness_started = time.perf_counter()

    log_event(
        logger,
        "agent1_harness_begin",
        step="agent1_begin",
        status="ok",
        data={
            "expected_trend": facts.expected_trend,
            "n_prices": len(facts.prices),
        },
    )

    for attempt in range(1, max_attempts + 1):
        prompt = build_performance_prompt(ticker, forecast_text, facts)
        if attempt > 1:
            prompt += (
                "\n\nPREVIOUS OUTPUT FAILED GUARDRAILS. "
                f"Errors: {', '.join(result.errors)}. "
                f"Respond again with Trend: {facts.expected_trend} exactly, "
                "and a medium-length Analysis (1–2 short paragraphs)."
            )
            log_event(
                logger,
                "agent1_retry",
                step="agent1_retry",
                status="retry",
                metrics={"attempt": attempt},
                data={"prior_errors": list(result.errors)},
            )
        else:
            log_event(
                logger,
                "agent1_llm_invoke",
                step="agent1_llm",
                status="started",
                metrics={"attempt": attempt},
            )

        invoke_started = time.perf_counter()
        response = call(
            [
                SystemMessage(
                    content=(
                        "You write compact equity research notes. "
                        "Stay medium length — useful, not long essays."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        invoke_ms = (time.perf_counter() - invoke_started) * 1000
        text = message_text(response)
        result = validate_performance_analysis(text, facts)
        attempts.append(
            {
                "attempt": attempt,
                "ok": result.ok,
                "errors": list(result.errors),
                "parsed_trend": result.parsed_trend,
                "chars": len(text or ""),
                "duration_ms": round(invoke_ms, 2),
            }
        )
        log_event(
            logger,
            "agent1_validation",
            step="agent1_validation",
            status="success" if result.ok else "failed",
            duration_ms=invoke_ms,
            metrics={
                "attempt": attempt,
                "chars": len(text or ""),
                "ok": result.ok,
            },
            data={
                "trend": result.parsed_trend,
                "errors": list(result.errors),
            },
        )
        if result.ok:
            break

    repaired = False
    if not result.ok:
        log_event(
            logger,
            "agent1_repair",
            level=logging.WARNING,
            step="agent1_repair",
            status="repair",
            metrics={"after_attempts": max_attempts},
        )
        text = repair_performance_analysis(text, facts, ticker)
        result = validate_performance_analysis(text, facts)
        repaired = True
        if not result.ok:
            log_event(
                logger,
                "agent1_deterministic_fallback",
                level=logging.WARNING,
                step="agent1_fallback",
                status="fallback",
            )
            text = deterministic_performance_analysis(facts, ticker)
            result = validate_performance_analysis(text, facts)

    total_ms = (time.perf_counter() - harness_started) * 1000
    log_event(
        logger,
        "agent1_harness_end",
        step="agent1_end",
        status="success" if result.ok else "failed",
        duration_ms=total_ms,
        metrics={
            "repaired": repaired,
            "chars": len(text or ""),
            "attempts": len(attempts),
        },
        data={"trend": facts.expected_trend, "guardrail_ok": result.ok},
    )

    return {
        "performance_analysis": text,
        "performance_trend": facts.expected_trend,
        "performance_guardrail_ok": result.ok,
        "performance_guardrail_errors": list(result.errors),
        "performance_repaired": repaired,
        "performance_attempts": attempts,
        "performance_duration_ms": round(total_ms, 2),
        "forecast_facts": {
            "expected_trend": facts.expected_trend,
            "low": facts.low,
            "high": facts.high,
            "n_prices": len(facts.prices),
        },
    }


CHIP_KEYS = ("trend", "news", "confidence", "projected_move", "risk")

_CHIP_SECTION_RE = re.compile(
    r"(?is)^\s*(trend|news|confidence|projected\s*move|risk)\s*:\s*(.+?)(?=^\s*(?:trend|news|confidence|projected\s*move|risk)\s*:|\Z)",
    re.MULTILINE,
)


def projected_move_pct(forecast: dict[str, Any] | None) -> float | None:
    """Signed % from last close to the final forecast session."""
    data = forecast or {}
    last_close = data.get("last_close")
    preds = data.get("predictions") or []
    if last_close is None or not preds:
        return None
    try:
        terminal = float(preds[-1].get("value"))
        base = float(last_close)
        if base == 0:
            return None
        return ((terminal / base) - 1.0) * 100.0
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return None


def format_projected_move(pct: float | None) -> str:
    if pct is None:
        return "—"
    return f"{pct:+.2f}%"


def deterministic_chip_explanations(
    *,
    ticker: str,
    trend: str | None,
    news_sentiment: str | None,
    confidence: str | None,
    projected_move: str,
    horizon: int | None = None,
    risk_level: str | None = None,
) -> dict[str, str]:
    """Plain-language fallbacks when the chip LLM step is skipped or fails."""
    t = (trend or "SIDEWAYS").upper()
    n = (news_sentiment or "MIXED").upper()
    c = (confidence or "Medium").strip() or "Medium"
    r = (risk_level or "MODERATE").upper()
    sessions = horizon or 5
    sym = (ticker or "this ticker").upper()

    if t == "BULLISH":
        trend_text = (
            f"Trend is BULLISH for {sym}: the near-term forecast path ends above the "
            f"last close, so the Performance Analyst labeled direction upward on this "
            f"short horizon. That is a path reading from the model sessions, not a "
            f"buy recommendation."
        )
    elif t == "BEARISH":
        trend_text = (
            f"Trend is BEARISH for {sym}: the near-term forecast path ends below the "
            f"last close, so the Performance Analyst labeled direction downward on this "
            f"short horizon. That is a path reading from the model sessions, not a "
            f"sell recommendation."
        )
    else:
        trend_text = (
            f"Trend is SIDEWAYS for {sym}: the forecast path stays close to the last "
            f"close over the short horizon, so the Performance Analyst did not call a "
            f"clear up or down move. Flat or choppy paths land here."
        )

    if n == "POSITIVE":
        news_text = (
            f"News is POSITIVE for {sym}: the Market Expert’s briefing leaned "
            f"constructive across the headlines it reviewed. Headline tone can change "
            f"quickly and does not override the forecast path."
        )
    elif n == "NEGATIVE":
        news_text = (
            f"News is NEGATIVE for {sym}: the Market Expert’s briefing leaned "
            f"cautious or adverse across the headlines it reviewed. Headline tone can "
            f"change quickly and does not override the forecast path."
        )
    else:
        news_text = (
            f"News is MIXED for {sym}: coverage pulled in more than one direction, or "
            f"there was not a clear single tone. The Market Expert treated sentiment as "
            f"balanced rather than strongly positive or negative."
        )

    if c.lower() == "low":
        conf_text = (
            f"Confidence is Low for this run because the path used a simpler fallback "
            f"or needed extra checking before the note was shown. Treat the summary "
            f"chips as a lighter signal and read the agent notes for caveats."
        )
    elif c.lower() == "high":
        conf_text = (
            f"Confidence is High for this run: the forecast path and Performance note "
            f"aligned without heavy repair. It still reflects a short research window, "
            f"not certainty about future prices."
        )
    else:
        conf_text = (
            f"Confidence is Medium for this run: a normal research-desk reading of the "
            f"forecast path with usual short-horizon uncertainty. Use it as context "
            f"alongside Trend and the agent notes, not as a guarantee."
        )

    move_text = (
        f"Projected move is {projected_move} for {sym}: the change from the last "
        f"close to the final forecast session across about {sessions} sessions. "
        f"It summarizes how far the path stretches, not a promised return."
    )

    if r == "ELEVATED":
        risk_text = (
            f"Risk is ELEVATED for {sym}: coded volatility, drawdown, news, financial, "
            f"or forecast-trust flags stacked higher on this run. Read the Risk analyst "
            f"note for which drivers mattered — it is not a trade instruction."
        )
    elif r == "CONTAINED":
        risk_text = (
            f"Risk is CONTAINED for {sym}: coded path volatility and context flags stayed "
            f"comparatively calm on this short window. Contained still means uncertainty "
            f"remains; it is not a guarantee of quiet markets."
        )
    else:
        risk_text = (
            f"Risk is MODERATE for {sym}: a normal short-horizon mix of path volatility "
            f"and agent context without an extreme stack of warning flags. Use it with "
            f"Trend, News, and Financial — not alone."
        )

    return {
        "trend": trend_text,
        "news": news_text,
        "confidence": conf_text,
        "projected_move": move_text,
        "risk": risk_text,
    }


def parse_chip_explanations(text: str) -> dict[str, str]:
    """Parse Trend:/News:/Confidence:/Projected move: sections from model output."""
    out: dict[str, str] = {}
    for match in _CHIP_SECTION_RE.finditer(text or ""):
        key = re.sub(r"\s+", "_", match.group(1).strip().lower())
        if key == "projected_move" or key in CHIP_KEYS:
            body = re.sub(r"\s+", " ", (match.group(2) or "").strip())
            if body:
                out[key if key in CHIP_KEYS else "projected_move"] = body
    return {k: out[k] for k in CHIP_KEYS if k in out}


def build_chip_explanation_prompt(
    *,
    ticker: str,
    trend: str,
    news_sentiment: str,
    confidence: str,
    projected_move: str,
    horizon: int,
    performance_analysis: str,
    news_summary: str,
    risk_level: str = "MODERATE",
    risk_analysis: str = "",
) -> str:
    perf = (performance_analysis or "").strip()
    if len(perf) > 1200:
        perf = perf[:1200] + "…"
    news = (news_summary or "").strip()
    if len(news) > 900:
        news = news[:900] + "…"
    risk_note = (risk_analysis or "").strip()
    if len(risk_note) > 900:
        risk_note = risk_note[:900] + "…"
    return f"""You are the Performance Analyst writing hover blurbs for the summary chips on a research desk for {ticker}.

LOCKED LABELS (do not change them):
- Trend: {trend}
- News: {news_sentiment}
- Confidence: {confidence}
- Projected move: {projected_move} (about {horizon} sessions)
- Risk: {risk_level}

PERFORMANCE NOTE (ground Trend / Confidence / Projected move here):
{perf or "(none)"}

NEWS BRIEFING (ground News here; do not invent headlines):
{news or "(none)"}

RISK NOTE (ground Risk here):
{risk_note or "(none)"}

Write EXACTLY five sections with these headers:

Trend:
<2–4 sentences explaining why this run’s Trend is {trend} from the forecast path. Do not explain the opposite label.>

News:
<2–4 sentences explaining why News is {news_sentiment} from the briefing. Do not invent articles.>

Confidence:
<2–3 sentences on what Confidence {confidence} means for this run’s path quality.>

Projected move:
<2–3 sentences on what {projected_move} means (last close → final forecast session).>

Risk:
<2–3 sentences on what Risk {risk_level} means for this run’s downside / uncertainty. No trade advice.>

Rules:
- Product language only — no model names, vendors, caches, or internals.
- Research support only — no buy/sell advice.
- Keep each section concise; say only what helps the reader understand the chip.
- Do not add extra sections or markdown bold.
"""


def explain_summary_chips(
    *,
    ticker: str,
    trend: str | None,
    news_sentiment: str | None,
    confidence: str | None,
    forecast: dict[str, Any] | None,
    performance_analysis: str | None = None,
    news_summary: str | None = None,
    risk_level: str | None = None,
    risk_analysis: str | None = None,
    llm: Any = None,
    invoke: Callable[..., Any] | None = None,
) -> dict[str, str]:
    """One Performance-owned LLM pass for the summary-chip blurbs.

    Fail-soft: always returns all chip keys (deterministic fallback if needed).
    """
    set_ticker(ticker)
    t = (trend or "SIDEWAYS").upper()
    n = (news_sentiment or "MIXED").upper()
    c = (confidence or "Medium").strip() or "Medium"
    r = (risk_level or "MODERATE").upper()
    data = forecast or {}
    horizon = int(data.get("horizon") or len(data.get("predictions") or []) or 5)
    pct = projected_move_pct(data)
    move = format_projected_move(pct)
    fallback = deterministic_chip_explanations(
        ticker=ticker,
        trend=t,
        news_sentiment=n,
        confidence=c,
        projected_move=move,
        horizon=horizon,
        risk_level=r,
    )

    if llm is None and invoke is None:
        return fallback

    call = invoke or (lambda messages: llm.invoke(messages))
    prompt = build_chip_explanation_prompt(
        ticker=ticker,
        trend=t,
        news_sentiment=n,
        confidence=c,
        projected_move=move,
        horizon=horizon,
        performance_analysis=performance_analysis or "",
        news_summary=news_summary or "",
        risk_level=r,
        risk_analysis=risk_analysis or "",
    )
    started = time.perf_counter()
    try:
        response = call(
            [
                SystemMessage(
                    content=(
                        "You write short chip explanations for an equity research desk. "
                        "Stay grounded in the provided labels and notes."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        text = message_text(response)
        parsed = parse_chip_explanations(text)
        merged = {**fallback, **parsed}
        log_event(
            logger,
            "agent1_chip_explain",
            step="agent1_chips",
            status="ok",
            duration_ms=(time.perf_counter() - started) * 1000,
            metrics={"parsed": len(parsed), "chars": len(text or "")},
            data={"trend": t, "news": n, "confidence": c, "risk": r},
        )
        return {k: merged[k] for k in CHIP_KEYS}
    except Exception as exc:  # noqa: BLE001 — never block analyze
        log_event(
            logger,
            "agent1_chip_explain",
            level=logging.WARNING,
            step="agent1_chips",
            status="fallback",
            duration_ms=(time.perf_counter() - started) * 1000,
            data={"error": str(exc)[:200]},
        )
        return fallback
