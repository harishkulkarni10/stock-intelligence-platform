"""Harness for the Financial Analyst (agent 3): facts, prompt, validate, repair.

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
from src.market.financials import HEALTH_LABELS

logger = get_logger("sip.agent3.harness")

MIN_ANALYSIS_CHARS = 180
_HEALTH_LINE_RE = re.compile(
    r"(?im)^\s*health\s*:\s*(strong|adequate|stressed|unavailable)\b"
)
_ANALYSIS_BLOCK_RE = re.compile(
    r"(?is)^\s*analysis\s*:\s*(.+?)(?=^\s*(?:strengths|weaknesses|caveats?|health)\s*:|\Z)",
    re.MULTILINE,
)
# Capture currency, plain numbers, and percents; skip years like 2024 when alone.
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])(?:\$)?(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+\.\d+|-?\d+)(?:\s*(?:B|M|K|%))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FinancialFacts:
    ticker: str
    status: str
    coverage: str
    allowed_health: str
    metrics: dict[str, float | None]
    signals: dict[str, str]
    allowed_numbers: tuple[float, ...]
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str = "USD"


@dataclass(frozen=True)
class FinancialGuardrailResult:
    ok: bool
    parsed_health: str | None
    errors: tuple[str, ...]


def parse_health_label(text: str) -> str | None:
    match = _HEALTH_LINE_RE.search(text or "")
    if not match:
        return None
    return match.group(1).upper()


def _analysis_body(text: str) -> str:
    match = _ANALYSIS_BLOCK_RE.search(text or "")
    if not match:
        return ""
    return match.group(1).strip()


def build_financial_facts(financials: dict[str, Any] | None, *, ticker: str = "") -> FinancialFacts:
    payload = financials or {}
    symbol = str(payload.get("ticker") or ticker or "").upper()
    status = str(payload.get("status") or "error")
    coverage = str(payload.get("coverage") or "missing")
    allowed_health = str(payload.get("allowed_health") or "UNAVAILABLE").upper()
    if allowed_health not in HEALTH_LABELS:
        allowed_health = "UNAVAILABLE"
    metrics_raw = payload.get("metrics") or {}
    metrics: dict[str, float | None] = {}
    for key, value in metrics_raw.items():
        try:
            metrics[str(key)] = None if value is None else float(value)
        except (TypeError, ValueError):
            metrics[str(key)] = None
    signals = {
        str(k): str(v)
        for k, v in (payload.get("signals") or {}).items()
    }
    allowed = tuple(
        float(x)
        for x in (payload.get("allowed_numbers") or [])
        if isinstance(x, (int, float)) and not math.isnan(float(x))
    )
    return FinancialFacts(
        ticker=symbol,
        status=status,
        coverage=coverage,
        allowed_health=allowed_health,
        metrics=metrics,
        signals=signals,
        allowed_numbers=allowed,
        name=(str(payload.get("name")).strip() if payload.get("name") else None),
        sector=(str(payload.get("sector")).strip() if payload.get("sector") else None),
        industry=(
            str(payload.get("industry")).strip() if payload.get("industry") else None
        ),
        currency=str(payload.get("currency") or "USD"),
    )


def _near_allowed(value: float, allowed: tuple[float, ...]) -> bool:
    if not allowed:
        return False
    for candidate in allowed:
        if candidate == 0:
            if abs(value) < 1e-6:
                return True
            continue
        # Absolute tolerance for small ratios; relative for large money figures.
        tol = max(0.05 * abs(candidate), 0.05 if abs(candidate) < 50 else 0.5)
        if abs(value - candidate) <= tol:
            return True
        # Scaled money: model wrote 12.3 for 12.3B while facts store raw.
        for scale in (1e6, 1e9, 1e3, 100.0, 0.01):
            if abs(value * scale - candidate) <= max(0.05 * abs(candidate), 0.5):
                return True
            if abs(value - candidate * scale) <= max(0.05 * abs(candidate * scale), 0.5):
                return True
    return False


def _extract_claim_numbers(text: str) -> list[float]:
    claims: list[float] = []
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(0)
        token = match.group(1).replace(",", "")
        try:
            number = float(token)
        except ValueError:
            continue
        # Skip bare years.
        if re.fullmatch(r"20\d{2}", token):
            continue
        upper = raw.upper()
        if upper.endswith("%"):
            claims.append(number)
            continue
        if upper.endswith("B"):
            claims.append(number * 1e9)
            claims.append(number)
            continue
        if upper.endswith("M"):
            claims.append(number * 1e6)
            claims.append(number)
            continue
        if upper.endswith("K"):
            claims.append(number * 1e3)
            claims.append(number)
            continue
        claims.append(number)
    return claims


def _invented_numbers(text: str, facts: FinancialFacts) -> bool:
    if facts.allowed_health == "UNAVAILABLE":
        # UNAVAILABLE path should not introduce concrete fundamentals.
        body = _HEALTH_LINE_RE.sub("", text or "")
        for claim in _extract_claim_numbers(body):
            if abs(claim) >= 10:  # ignore tiny counters / list indices
                return True
        return False
    if not facts.allowed_numbers:
        return False
    analysis = _analysis_body(text) or (text or "")
    for claim in _extract_claim_numbers(analysis):
        if abs(claim) < 1.0:
            continue
        if not _near_allowed(claim, facts.allowed_numbers):
            return True
    return False


def validate_financial_analysis(text: str, facts: FinancialFacts) -> FinancialGuardrailResult:
    errors: list[str] = []
    parsed = parse_health_label(text)
    if parsed is None:
        errors.append("missing_health_label")
    elif parsed not in HEALTH_LABELS:
        errors.append(f"invalid_health:{parsed}")
    elif parsed != facts.allowed_health:
        errors.append(f"health_mismatch:got={parsed}:expected={facts.allowed_health}")

    if facts.allowed_health == "UNAVAILABLE":
        if parsed and parsed != "UNAVAILABLE":
            errors.append("health_must_be_unavailable")
    else:
        if not re.search(r"(?im)^\s*analysis\s*:", text or ""):
            errors.append("missing_analysis_section")
        else:
            body = _analysis_body(text)
            if len(body) < MIN_ANALYSIS_CHARS:
                errors.append(f"analysis_too_short:{len(body)}<{MIN_ANALYSIS_CHARS}")
        if not re.search(r"(?im)^\s*strengths\s*:", text or ""):
            errors.append("missing_strengths_section")
        if not re.search(r"(?im)^\s*weaknesses\s*:", text or ""):
            errors.append("missing_weaknesses_section")
        if not re.search(r"(?im)^\s*caveats?\s*:", text or ""):
            errors.append("missing_caveats_section")
        if _invented_numbers(text, facts):
            errors.append("invented_or_ungrounded_number")

    if facts.allowed_health == "UNAVAILABLE" and _invented_numbers(text, facts):
        errors.append("invented_numbers_on_unavailable")

    return FinancialGuardrailResult(ok=not errors, parsed_health=parsed, errors=tuple(errors))


def deterministic_financial_analysis(facts: FinancialFacts) -> str:
    if facts.allowed_health == "UNAVAILABLE":
        return (
            "Health: UNAVAILABLE\n"
            "Analysis:\n"
            "Usable fundamental fields were missing or failed to load for this ticker, "
            "so a grounded financial view cannot be formed without inventing statement "
            "figures. Treat fundamentals as missing for this run and rely on primary "
            "filings or a data vendor you trust before drawing business-quality conclusions.\n"
            "Strengths:\n"
            "- None established from the available snapshot.\n"
            "Weaknesses:\n"
            "- Incomplete or missing fundamental coverage.\n"
            "Caveats: Do not invent revenue, margins, debt, or valuation multiples."
        )

    m = facts.metrics
    signals = facts.signals

    def money(key: str) -> str:
        value = m.get(key)
        if value is None:
            return "n/a"
        abs_n = abs(value)
        if abs_n >= 1e9:
            return f"{value / 1e9:.2f}B"
        if abs_n >= 1e6:
            return f"{value / 1e6:.2f}M"
        return f"{value:.2f}"

    def pct(key: str) -> str:
        value = m.get(key)
        return "n/a" if value is None else f"{value:.1f}%"

    analysis = (
        f"Based on the fetched {facts.coverage} fundamental snapshot for {facts.ticker}, "
        f"code-assigned health is {facts.allowed_health}. "
        f"Revenue is {money('revenue')} with YoY {pct('revenue_yoy_pct')}; "
        f"net margin {pct('net_margin_pct')}, free cash flow {money('free_cashflow')}, "
        f"and total debt {money('total_debt')}. "
        f"Profitability signal={signals.get('profitability', 'unknown')}, "
        f"leverage={signals.get('leverage', 'unknown')}, "
        f"valuation heuristic={signals.get('valuation', 'unknown')} "
        f"(PE {m.get('pe') if m.get('pe') is not None else 'n/a'}). "
        "This note only restates those fields; gaps mean limited confidence, not a hidden thesis."
    )
    # Ensure min length without inventing numbers.
    if len(analysis) < MIN_ANALYSIS_CHARS:
        analysis += (
            " Use the metrics block as the only numeric source and treat missing fields "
            "as unknown rather than filling them with sector folklore."
        )

    strengths = []
    weaknesses = []
    if signals.get("profitability") in {"improving", "stable"}:
        strengths.append("- Profitability signal is not weak on the available margins/FCF.")
    else:
        weaknesses.append("- Profitability looks weak or unknown on the available fields.")
    if signals.get("leverage") == "conservative":
        strengths.append("- Leverage signal reads conservative on the debt markers present.")
    elif signals.get("leverage") == "stretched":
        weaknesses.append("- Leverage signal is stretched on debt/liquidity markers.")
    else:
        weaknesses.append("- Leverage picture is only moderate or incomplete.")
    if signals.get("valuation") == "cheap":
        strengths.append("- Valuation heuristic is not rich on the multiples present.")
    elif signals.get("valuation") == "rich":
        weaknesses.append("- Valuation heuristic screens rich on the multiples present.")
    if not strengths:
        strengths.append("- Coverage is partial; no clean strength established.")
    if not weaknesses:
        weaknesses.append("- No single weakness dominated the coded signals.")

    return (
        f"Health: {facts.allowed_health}\n"
        f"Analysis:\n{analysis}\n"
        "Strengths:\n"
        + "\n".join(strengths)
        + "\nWeaknesses:\n"
        + "\n".join(weaknesses)
        + "\nCaveats: Yahoo fundamentals can be stale or incomplete; valuation bands are heuristics only."
    )


def repair_financial_analysis(text: str, facts: FinancialFacts) -> str:
    if facts.allowed_health == "UNAVAILABLE":
        return deterministic_financial_analysis(facts)
    body = _HEALTH_LINE_RE.sub("", (text or "").strip()).strip()
    if not body:
        return deterministic_financial_analysis(facts)
    if not re.search(r"(?im)^\s*analysis\s*:", body):
        body = f"Analysis:\n{body}"
    if not re.search(r"(?im)^\s*strengths\s*:", body):
        body = f"{body}\nStrengths:\n- See metrics snapshot."
    if not re.search(r"(?im)^\s*weaknesses\s*:", body):
        body = f"{body}\nWeaknesses:\n- See metrics snapshot and coverage gaps."
    if not re.search(r"(?im)^\s*caveats?\s*:", body):
        body = f"{body}\nCaveats: Fundamentals may be incomplete or delayed."
    repaired = f"Health: {facts.allowed_health}\n{body}".strip()
    if validate_financial_analysis(repaired, facts).ok:
        return repaired
    return deterministic_financial_analysis(facts)


def build_financial_prompt(ticker: str, financials_raw: str, facts: FinancialFacts) -> str:
    if facts.allowed_health == "UNAVAILABLE":
        return f"""You are a Financial Analyst for {ticker}.

FUNDAMENTALS:
{financials_raw}

There are no usable fundamentals. Write EXACTLY:

Health: UNAVAILABLE
Analysis:
Usable fundamental fields were missing, so a grounded financial view cannot be formed without inventing figures. Explain briefly why inventing statements would be harmful.
Strengths:
- None established from the available snapshot.
Weaknesses:
- Incomplete or missing fundamental coverage.
Caveats: Do not invent revenue, margins, debt, or valuation multiples.

Rules:
- Do not invent any financial figures.
- Do not discuss price forecasts or news headlines.
"""

    return f"""You are a Financial Analyst. Write a compact fundamental note for {ticker} — clear and useful, not an essay.

FUNDAMENTALS (code-fetched; only source of numbers):
{financials_raw}

Required health label from code: {facts.allowed_health}

Use EXACTLY this structure:

Health: {facts.allowed_health}
Analysis:
<1–2 short paragraphs (about 80–140 words). Interpret profitability, cash, leverage, and valuation using ONLY the metrics above. Mention coverage gaps. Do not invent figures.>
Strengths:
- <grounded bullet>
- <optional grounded bullet>
Weaknesses:
- <grounded bullet>
- <optional grounded bullet>
Caveats:
<1–2 sentences on data freshness, missing fields, or heuristic valuation bands>

Rules:
- Health line MUST be exactly: {facts.allowed_health}
- Medium length only — no long essays.
- Vary wording from run to run; keep the same facts and Health label.
- Use ONLY numbers present in FUNDAMENTALS.
- No buy/sell recommendation.
- Do not discuss price forecasts or news headlines.
"""


def run_financial_harness(
    *,
    ticker: str,
    financials: dict[str, Any] | None,
    financials_raw: str,
    llm: Any,
    invoke: Callable[..., Any] | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Prompt → validate → retry → repair/fallback."""
    set_ticker(ticker)
    facts = build_financial_facts(financials, ticker=ticker)
    call = invoke or (lambda messages: llm.invoke(messages))
    attempts: list[dict[str, Any]] = []
    text = ""
    result = FinancialGuardrailResult(ok=False, parsed_health=None, errors=("not_run",))
    harness_started = time.perf_counter()

    log_event(
        logger,
        "agent3_harness_begin",
        step="agent3_begin",
        status="ok",
        data={
            "status": facts.status,
            "coverage": facts.coverage,
            "allowed_health": facts.allowed_health,
        },
    )

    for attempt in range(1, max_attempts + 1):
        prompt = build_financial_prompt(ticker, financials_raw, facts)
        if attempt > 1:
            prompt += (
                "\n\nPREVIOUS OUTPUT FAILED GUARDRAILS. "
                f"Errors: {', '.join(result.errors)}. "
                f"Respond again with Health: {facts.allowed_health} and only FUNDAMENTALS numbers."
            )
            log_event(
                logger,
                "agent3_retry",
                step="agent3_retry",
                status="retry",
                metrics={"attempt": attempt},
                data={"prior_errors": list(result.errors)},
            )
        else:
            log_event(
                logger,
                "agent3_llm_invoke",
                step="agent3_llm",
                status="started",
                metrics={"attempt": attempt},
            )

        invoke_started = time.perf_counter()
        response = call(
            [
                SystemMessage(
                    content=(
                        "You write compact fundamental equity notes. "
                        "Stay medium length and never invent financial figures."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        invoke_ms = (time.perf_counter() - invoke_started) * 1000
        text = message_text(response)
        result = validate_financial_analysis(text, facts)
        attempts.append(
            {
                "attempt": attempt,
                "ok": result.ok,
                "errors": list(result.errors),
                "parsed_health": result.parsed_health,
                "chars": len(text or ""),
                "duration_ms": round(invoke_ms, 2),
            }
        )
        log_event(
            logger,
            "agent3_validation",
            step="agent3_validation",
            status="success" if result.ok else "failed",
            duration_ms=invoke_ms,
            metrics={"attempt": attempt, "chars": len(text or ""), "ok": result.ok},
            data={"health": result.parsed_health, "errors": list(result.errors)},
        )
        if result.ok:
            break

    repaired = False
    if not result.ok:
        log_event(
            logger,
            "agent3_repair",
            level=logging.WARNING,
            step="agent3_repair",
            status="repair",
            metrics={"after_attempts": max_attempts},
        )
        text = repair_financial_analysis(text, facts)
        result = validate_financial_analysis(text, facts)
        repaired = True
        if not result.ok:
            log_event(
                logger,
                "agent3_deterministic_fallback",
                level=logging.WARNING,
                step="agent3_fallback",
                status="fallback",
            )
            text = deterministic_financial_analysis(facts)
            result = validate_financial_analysis(text, facts)

    total_ms = (time.perf_counter() - harness_started) * 1000
    log_event(
        logger,
        "agent3_harness_end",
        step="agent3_end",
        status="success" if result.ok else "failed",
        duration_ms=total_ms,
        metrics={
            "repaired": repaired,
            "chars": len(text or ""),
            "attempts": len(attempts),
        },
        data={"health": result.parsed_health, "guardrail_ok": result.ok},
    )

    return {
        "financial_analysis": text,
        "financial_health": result.parsed_health or facts.allowed_health,
        "financial_guardrail_ok": result.ok,
        "financial_guardrail_errors": list(result.errors),
        "financial_repaired": repaired,
        "financial_attempts": attempts,
        "financial_duration_ms": round(total_ms, 2),
        "financial_facts": {
            "status": facts.status,
            "coverage": facts.coverage,
            "allowed_health": facts.allowed_health,
        },
    }
