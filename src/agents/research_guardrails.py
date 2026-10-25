"""Grounding harness for Research Co-Pilot — facts, validate, repair, fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agents.performance_guardrails import format_projected_move, projected_move_pct
from src.agents.research_tools import (
    DeskContext,
    compare_labels,
    explain_confidence,
    get_forecast_path,
    get_risk_metrics,
    get_run_overview,
)

_DEFAULT_NEXT = (
    "NEXT: Explain disagreement|Walk through Risk|Why this confidence?|Forecast path"
)
OUT_OF_SCOPE = (
    "Out of scope — Research Co-Pilot explains finished desk packets only. "
    "It does not give buy/sell advice, pick tickers, or answer general trivia "
    "unrelated to the open analysis."
)
ADVICE_META = (
    "### Research support only\n"
    "This desk explains labels, disagreement, forecast path, and Risk from "
    "finished run packets. It is not a broker and does not give buy, sell, "
    "or hold advice.\n\n"
    "### What to ask instead\n"
    "- Why this confidence?\n"
    "- Explain disagreement\n"
    "- Walk through Risk\n"
    "- Forecast path / projected move\n\n"
    f"{_DEFAULT_NEXT}"
)
_ADVICE_META_RE = re.compile(
    r"(?i)\b("
    r"why (?:don'?t|do not|can'?t|cannot|wont|won'?t) you.{0,60}"
    r"(?:buy|sell|advice|recommend)|"
    r"why (?:no|not).{0,40}(?:buy|sell|advice)|"
    r"why (?:won'?t|dont).{0,40}(?:advise|recommend)"
    r")\b"
)

_TICKER_TOKEN_RE = re.compile(r"\b([A-Za-z]{1,5})\b")
_TICKER_STOPWORDS = frozenset(
    {
        "A",
        "I",
        "AM",
        "AN",
        "AS",
        "AT",
        "BE",
        "BY",
        "DO",
        "GO",
        "IF",
        "IN",
        "IS",
        "IT",
        "ME",
        "MY",
        "NO",
        "OF",
        "ON",
        "OR",
        "SO",
        "TO",
        "UP",
        "US",
        "WE",
        "AI",
        "CEO",
        "CFO",
        "CTO",
        "ETF",
        "IPO",
        "GDP",
        "FAQ",
        "API",
        "LLM",
        "NEXT",
        "RISK",
        "NEWS",
        "MIXED",
        "LOW",
        "HIGH",
        "TREND",
        "MOVE",
        "PATH",
        "RUN",
        "DESK",
        "OPEN",
        "THIS",
        "THAT",
        "WHAT",
        "WHEN",
        "WHY",
        "HOW",
        "WHO",
        "YES",
        "NOT",
        "ALSO",
        "INFO",
        "DATA",
        "PACKET",
        "SCOPE",
        "ONLY",
        "NOW",
        "CAN",
        "YOU",
        "GIVE",
        "TELL",
        "MEAN",
        "CHECK",
        "PROPERLY",
        "ABOUT",
        "FROM",
        "WITH",
        "YOUR",
        "HERE",
        "TALK",
        "KNOW",
        "SESSION",
        "ANALYSIS",
        "RESEARCH",
        "GUIDE",
        "COPILOT",
        "STRONG",
        "WEAK",
        "FLAT",
        "BULL",
        "BEAR",
    }
)
_FOREIGN_CONTEXT_RE = re.compile(
    r"(?i)\b("
    r"also in (?:the )?session|"
    r"also on (?:the )?desk|"
    r"other ticker|"
    r"previous (?:run|ticker)|"
    r"earlier (?:run|ticker)|"
    r"info about|"
    r"information about|"
    r"data (?:about|for|on)|"
    r"facts (?:about|for|on)|"
    r"tell me about|"
    r"what about|"
    r"check .{0,20}properly|"
    r"can\s*(?:not|'t)?\s+(?:give|talk|answer).{0,40}about"
    r")\b"
)

NO_RUN = (
    "### No open run\n"
    "Research mode needs a finished analysis on the desk.\n\n"
    "### What to do\n"
    "- Search a ticker and run agents first\n"
    "- Then ask about labels, disagreement, forecast path, or Risk on that run\n\n"
    "### Tip\n"
    "Switch to Guide for product how-to questions anytime.\n\n"
    "NEXT: How to read a run|What this desk is"
)

_ADVICE_RE = re.compile(
    r"(?i)\b(buy|sell|trim(?:\s+the)?\s+position|overweight|underweight|"
    r"go long|go short|accumulate shares|should you (?:buy|sell))\b"
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9.])(-?\d+(?:\.\d+)?)(?:\s*(?:%|pct|percent))?",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
_ADVICE_QUESTION_RE = re.compile(
    r"(?i)\b("
    r"should i (?:buy|sell|hold|trim)|"
    r"buy or sell|"
    r"is it a good (?:buy|time to buy)|"
    r"pick (?:a |me )?(?:stock|ticker)|"
    r"recommend a (?:stock|ticker)|"
    r"what should i (?:buy|sell|invest)"
    r")\b"
)


@dataclass(frozen=True)
class ResearchFacts:
    ticker: str
    performance_trend: str
    news_sentiment: str
    financial_health: str
    risk_level: str
    stance: str
    confidence: str
    projected_move_pct: float | None
    allowed_labels: tuple[str, ...]
    allowed_numbers: tuple[float, ...]
    allowed_urls: tuple[str, ...]


@dataclass(frozen=True)
class ResearchGuardrailResult:
    ok: bool
    errors: tuple[str, ...]


def is_advice_question(message: str) -> bool:
    return bool(_ADVICE_QUESTION_RE.search(message or ""))


def is_advice_meta_question(message: str) -> bool:
    """Why-no-advice questions get a policy explanation, not the blunt OOS brick."""
    return bool(_ADVICE_META_RE.search(message or ""))


def is_open_run_meta_question(message: str) -> bool:
    """Identity / binding questions — keep on scripted open-run copy."""
    q = (message or "").strip().lower()
    if not q:
        return False
    keys = (
        "why am i talking",
        "why am i here",
        "why i am here",
        "why i am talking",
        "how do you know",
        "which ticker",
        "which run",
        "what ticker",
        "current ticker",
        "open run",
        "open packet",
        "why this ticker",
        "who are you talking",
        "what run is this",
        "which stock",
    )
    return any(k in q for k in keys)


def desk_tickers_from_packet(packet: dict[str, Any] | None) -> list[str]:
    raw = (packet or {}).get("desk_tickers") or []
    out: list[str] = []
    for item in raw:
        sym = str(item or "").strip().upper()
        if sym and sym not in out:
            out.append(sym)
    return out


def foreign_tickers_in_question(
    message: str,
    open_ticker: str,
    desk_tickers: list[str] | None = None,
) -> list[str]:
    """Tickers the user is asking about that are not the open packet."""
    open_t = str(open_ticker or "").strip().upper()
    if not open_t:
        return []
    q = message or ""
    found: list[str] = []

    for sym in desk_tickers or []:
        other = str(sym or "").strip().upper()
        if not other or other == open_t:
            continue
        if re.search(rf"\b{re.escape(other)}\b", q, re.IGNORECASE):
            if other not in found:
                found.append(other)

    # Also catch explicit other-ticker asks when desk list is empty/missing.
    if not found and _FOREIGN_CONTEXT_RE.search(q):
        for match in _TICKER_TOKEN_RE.finditer(q):
            token = match.group(1).upper()
            if token == open_t or token in _TICKER_STOPWORDS:
                continue
            if len(token) < 2:
                continue
            if token not in found:
                found.append(token)

    return found


def missing_desk_ticker_reply(
    *,
    active_ticker: str,
    desk_tickers: list[str],
    mentioned: list[str],
) -> str:
    """User named a ticker that is not among finished desk packets."""
    other = mentioned[0] if mentioned else "that ticker"
    active = str(active_ticker or "—").upper()
    on_desk = ", ".join(desk_tickers) if desk_tickers else "(none yet)"
    return (
        f"### Not on this desk\n"
        f"{other} is not among the finished runs Research can see right now.\n\n"
        f"### Desk packets\n"
        f"- Active view: {active}\n"
        f"- Available: {on_desk}\n\n"
        f"### Tip\n"
        f"Run {other} on the desk first. Research can then answer about it "
        f"without you hunting it in the list.\n\n"
        f"{_DEFAULT_NEXT}"
    )


def foreign_ticker_reply(
    packet: dict[str, Any],
    mentioned: list[str],
) -> str:
    """Legacy helper — prefer missing_desk_ticker_reply with full DeskContext."""
    overview = get_run_overview(packet)
    return missing_desk_ticker_reply(
        active_ticker=str(overview.get("ticker") or ""),
        desk_tickers=desk_tickers_from_packet(packet),
        mentioned=mentioned,
    )


def open_run_meta_reply(
    packet: dict[str, Any],
    *,
    desk_tickers: list[str] | None = None,
    active_ticker: str | None = None,
) -> str:
    overview = get_run_overview(packet)
    move = overview.get("projected_move") or "—"
    desk = desk_tickers or desk_tickers_from_packet(packet)
    active = str(active_ticker or overview.get("ticker") or "").upper()
    target = str(overview.get("ticker") or active).upper()
    others = [t for t in desk if t != active]
    if len(desk) > 1:
        desk_line = (
            f"- Finished runs in Research scope: {', '.join(desk)}\n"
            f"- Name a ticker (or company) to target that packet — no need to "
            f"open it on screen"
        )
        tip = (
            f"Ask about any desk ticker directly (e.g. Why was {others[0]} "
            f"confidence Medium?). Unnamed questions default to the active view "
            f"({active})."
        )
    else:
        desk_line = f"- Only {target} is on the desk right now"
        tip = "Run more tickers to grow Research’s desk scope for this visit."
    return (
        f"### Desk Research scope\n"
        f"Research can answer about every finished run on this desk visit. "
        f"Active results view is {active}.\n\n"
        f"### Bound facts ({target})\n"
        f"- Stance {overview.get('stance')}; trend {overview.get('performance_trend')}\n"
        f"- Projected move {move}; confidence {overview.get('confidence')}\n"
        f"{desk_line}\n\n"
        f"### Tip\n"
        f"{tip}\n\n"
        f"{_DEFAULT_NEXT}"
    )


def _number_variants(value: float) -> list[float]:
    """Allow common LLM rewrites (billions, percents as 28 vs 0.28, etc.)."""
    n = float(value)
    out = [round(n, 4), round(abs(n), 4)]
    for div in (1_000_000_000.0, 1_000_000.0, 1_000.0, 100.0):
        if abs(n) >= div * 0.5:
            scaled = n / div
            out.append(round(scaled, 4))
            out.append(round(abs(scaled), 4))
    if abs(n) <= 2.0:
        out.append(round(n * 100.0, 4))
        out.append(round(abs(n) * 100.0, 4))
    if 1.0 < abs(n) <= 200.0:
        out.append(round(n / 100.0, 4))
        out.append(round(abs(n) / 100.0, 4))
    return out


def _collect_text_numbers(*chunks: Any) -> list[float]:
    found: list[float] = []
    for chunk in chunks:
        text = str(chunk or "")
        for match in _NUMBER_RE.finditer(text):
            try:
                found.extend(_number_variants(float(match.group(1))))
            except ValueError:
                continue
    return found


def build_research_facts(packet: dict[str, Any]) -> ResearchFacts:
    from src.agents.research_tools import _forecast_blob

    forecast = _forecast_blob(packet)
    move = projected_move_pct(forecast)

    labels = [
        str(packet.get("performance_trend") or "").upper(),
        str(packet.get("news_sentiment") or "").upper(),
        str(packet.get("financial_health") or "").upper(),
        str(packet.get("risk_level") or "").upper(),
        str(packet.get("recommendation") or "").upper(),
        str(packet.get("confidence") or ""),
    ]
    labels = [x for x in labels if x and x != "NONE"]

    numbers: list[float] = []
    if move is not None:
        numbers.extend(_number_variants(float(move)))
    last_close = forecast.get("last_close")
    if last_close is not None:
        try:
            numbers.extend(_number_variants(float(last_close)))
        except (TypeError, ValueError):
            pass
    for point in forecast.get("predictions") or []:
        if isinstance(point, dict) and point.get("value") is not None:
            try:
                numbers.extend(_number_variants(float(point["value"])))
            except (TypeError, ValueError):
                pass

    financials = packet.get("financials") or {}
    metrics = financials.get("metrics") if isinstance(financials, dict) else {}
    if isinstance(metrics, dict):
        for value in metrics.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numbers.extend(_number_variants(float(value)))
            elif isinstance(value, str):
                try:
                    numbers.extend(
                        _number_variants(float(value.replace("%", "").strip()))
                    )
                except ValueError:
                    pass

    risk = packet.get("risk") or {}
    risk_metrics = risk.get("metrics") if isinstance(risk, dict) else {}
    if isinstance(risk_metrics, dict):
        for value in risk_metrics.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numbers.extend(_number_variants(float(value)))

    # Pull figures already written into specialist notes (grounding allow-list).
    numbers.extend(
        _collect_text_numbers(
            packet.get("performance_analysis"),
            packet.get("news_summary"),
            packet.get("financial_analysis"),
            packet.get("risk_analysis"),
            packet.get("final_report"),
        )
    )

    urls: list[str] = []
    news = packet.get("news") or {}
    articles = news.get("articles") if isinstance(news, dict) else []
    for item in articles or []:
        if isinstance(item, dict):
            url = (item.get("url") or "").strip()
            if url:
                urls.append(url)

    # Common harmless integers (session counts, horizons).
    numbers.extend([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0])

    return ResearchFacts(
        ticker=str(packet.get("ticker") or "").upper(),
        performance_trend=str(packet.get("performance_trend") or "—").upper(),
        news_sentiment=str(packet.get("news_sentiment") or "—").upper(),
        financial_health=str(packet.get("financial_health") or "—").upper(),
        risk_level=str(packet.get("risk_level") or "—").upper(),
        stance=str(packet.get("recommendation") or "—").upper(),
        confidence=str(packet.get("confidence") or "—"),
        projected_move_pct=move,
        allowed_labels=tuple(dict.fromkeys(labels)),
        allowed_numbers=tuple(dict.fromkeys(numbers)),
        allowed_urls=tuple(dict.fromkeys(urls)),
    )


def _number_allowed(value: float, allowed: tuple[float, ...]) -> bool:
    for candidate in allowed:
        if abs(candidate) < 1e-9 and abs(value) < 1e-9:
            return True
        # Relative tolerance for prices / percents already in the packet.
        scale = max(abs(candidate), abs(value), 1.0)
        if abs(candidate - value) <= max(0.05, 0.02 * scale):
            return True
    return False


def validate_research_reply(text: str, facts: ResearchFacts) -> ResearchGuardrailResult:
    errors: list[str] = []
    body = (text or "").strip()
    if not body:
        return ResearchGuardrailResult(ok=False, errors=("empty_reply",))

    if body.lower().startswith("out of scope"):
        return ResearchGuardrailResult(ok=True, errors=())

    if _ADVICE_RE.search(body):
        errors.append("advice_language")

    for match in _NUMBER_RE.finditer(body):
        raw = match.group(1)
        try:
            value = float(raw)
        except ValueError:
            continue
        # Years must appear in packet notes / figures — blocks invented 2027-style claims.
        if value.is_integer() and 1900 <= value <= 2100:
            if not _number_allowed(value, facts.allowed_numbers):
                errors.append(f"unknown_number:{raw}")
            continue
        if abs(value) < 0.001:
            continue
        if not _number_allowed(value, facts.allowed_numbers):
            errors.append(f"unknown_number:{raw}")

    for url in _URL_RE.findall(body):
        if url.rstrip(".,);") not in facts.allowed_urls and url not in facts.allowed_urls:
            # Allow truncated URL prefixes that still match a packet source.
            if not any(u.startswith(url.rstrip(".,);")) or url.startswith(u) for u in facts.allowed_urls):
                errors.append(f"unknown_url:{url[:80]}")

    return ResearchGuardrailResult(ok=not errors, errors=tuple(errors))


def strip_advice_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    kept = [p for p in parts if p and not _ADVICE_RE.search(p)]
    return " ".join(kept).strip()


_TOOL_NAME_RE = re.compile(
    r"\(?\b(?:get_run_overview|explain_confidence|get_agent_note|get_forecast_path|"
    r"get_sources|get_financial_snapshot|get_risk_metrics|compare_labels)"
    r"\b(?:\([^)]*\))?\)?",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"\[packet figure\]", re.IGNORECASE)
_INTRO_RE = re.compile(
    r"(?im)^\s*("
    r"i am research co-?pilot.*"
    r"|i(?:'| a)?m (?:the )?research co-?pilot.*"
    r"|as (?:your )?research co-?pilot.*"
    r"|i answer questions (?:regarding|about).*"
    r"|this desk follows the performance trend.*"
    r"|i am designed to explain.*"
    r"|analysis scope\s*$"
    r"|analysis purpose\s*$"
    r"|current (?:session|context)(?: details)?\s*$"
    r"|data (?:limitations|availability)\s*$"
    r"|this session (?:is|provides|dedicated).*"
    r"|you are interacting with the research co-?pilot.*"
    r").*\n?"
)


def strip_tool_name_leaks(text: str) -> str:
    cleaned = _TOOL_NAME_RE.sub("", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)
    return cleaned.strip()


def strip_brochure_voice(text: str) -> str:
    """Remove self-intros and meta scope paragraphs."""
    cleaned = _INTRO_RE.sub("", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def ensure_next_chips(text: str) -> str:
    body = (text or "").strip()
    if not body:
        return body
    if re.search(r"(?im)^NEXT:\s*", body):
        return body
    return f"{body}\n\n{_DEFAULT_NEXT}"


def prefers_deterministic_answer(question: str) -> bool:
    """Label-glossary / confidence / stance-why stay on scripted copy for Phase 1."""
    if is_open_run_meta_question(question):
        return True
    q = (question or "").lower()
    keys = (
        "why this confidence",
        "why confidence",
        "what does confidence",
        "confidence mean",
        "why medium",
        "why low confidence",
        "why high confidence",
        "what does risk",
        "what does moderate",
        "what does mixed",
        "what does bullish",
        "what does bearish",
        "what does neutral",
        "what does trend",
        "projected move",
        "what do these labels",
        "what does it mean when",
        "metric definition",
        "explain the labels",
        "explain these metrics",
        "why is it bullish",
        "why is it bearish",
        "why is it neutral",
        "why bullish",
        "why bearish",
        "why neutral",
        "why was",
        "why is ko",
        "why is jpm",
    )
    if any(k in q for k in keys):
        return True
    if re.search(r"\bwhy\s+(?:is|was)\s+\w+\s+(?:bullish|bearish|neutral)\b", q):
        return True
    if "confidence" in q and any(
        w in q for w in ("why", "how sure", "mean", "explain", "driver")
    ):
        return True
    return False


def focused_metric_key(question: str) -> str | None:
    """When set, answer that metric's definition — not a full label dump."""
    q = (question or "").lower()
    if re.search(r"\bcompare\b", q) or re.search(r"\bvs\b|\bversus\b", q):
        return None
    hits: list[str] = []
    if re.search(r"\bmixed\b", q) and re.search(
        r"\bnews\b|\bsentiment\b|\bmean\b", q
    ):
        hits.append("news")
    elif re.search(r"what does it mean when news", q) or re.search(
        r"what does (?:mixed|positive|negative)(?:\s+news)?\s+mean", q
    ):
        hits.append("news")
    if re.search(r"\bprojected move\b", q):
        hits.append("projected_move")
    if re.search(r"what does (?:bullish|bearish|neutral|sideways)\b", q) or (
        ("what does" in q or "mean when" in q) and "trend" in q
    ):
        hits.append("trend")
    if re.search(r"what does (?:contained|elevated|moderate)\b", q) or (
        ("what does" in q or "mean when" in q) and "risk" in q
    ):
        hits.append("risk")
    if re.search(r"what does (?:strong|adequate|stressed)\b", q) or (
        ("what does" in q or "mean when" in q) and "financial" in q
    ):
        hits.append("financials")
    if "what does confidence" in q or "confidence mean" in q:
        hits.append("confidence")
    # One clear metric → focused; several mentioned → full glossary.
    uniq = list(dict.fromkeys(hits))
    return uniq[0] if len(uniq) == 1 else None


def is_compare_moves_question(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(r"\bcompare\b", q)
        and re.search(r"\b(projected move|forecast|move|path)\b", q)
    )


def is_multi_risk_question(question: str) -> bool:
    q = (question or "").lower()
    return "risk" in q and bool(
        re.search(r"\bcompare\b|,|\band\b|\bversus\b|\bvs\b", q)
    )


def strip_unknown_number_sentences(text: str, facts: ResearchFacts) -> str:
    """Drop sentences that invent numbers rather than inserting placeholders."""
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    kept: list[str] = []
    for part in parts:
        if not part:
            continue
        bad = False
        for match in _NUMBER_RE.finditer(part):
            raw = match.group(1)
            try:
                value = float(raw)
            except ValueError:
                continue
            # Years only allowed if they appear in packet-derived allowed numbers.
            if value.is_integer() and 1900 <= value <= 2100:
                if not _number_allowed(value, facts.allowed_numbers):
                    bad = True
                    break
                continue
            if abs(value) < 0.001:
                continue
            if not _number_allowed(value, facts.allowed_numbers):
                bad = True
                break
        if not bad:
            kept.append(part)
    return " ".join(kept).strip()


_CATALYST_CLAIM_RE = re.compile(
    r"(?i)\b("
    r"partnership|partner(?:ing|ed)? with|acquisition|merger|takeover|"
    r"deal with|upcoming (?:deal|launch|partnership)|kfc|mcdonald|"
    r"contract with|agreement with"
    r")\b"
)


def strip_ungrounded_catalyst_sentences(text: str, packet: dict[str, Any]) -> str:
    """Drop partnership/deal claims unless that wording already exists in the packet."""
    blob = " ".join(
        str(packet.get(k) or "")
        for k in (
            "performance_analysis",
            "news_summary",
            "financial_analysis",
            "risk_analysis",
            "final_report",
        )
    ).lower()
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    kept: list[str] = []
    for part in parts:
        if not part:
            continue
        if _CATALYST_CLAIM_RE.search(part):
            # Require a distinctive 4+ letter match from the claim to appear in notes.
            tokens = [
                t.lower()
                for t in re.findall(r"[A-Za-z]{4,}", part)
                if t.lower()
                not in {
                    "with",
                    "this",
                    "that",
                    "from",
                    "long",
                    "term",
                    "major",
                    "upcoming",
                    "partnership",
                    "partner",
                    "bolstered",
                    "sentiment",
                    "growth",
                    "catalyst",
                    "catalysts",
                }
            ]
            if not tokens or not any(t in blob for t in tokens):
                continue
        kept.append(part)
    return " ".join(kept).strip()


def polish_research_markdown(text: str) -> str:
    """Fix glued headings / Tip lines that models sometimes emit."""
    cleaned = text or ""
    cleaned = re.sub(r"([^\n#])###\s*", r"\1\n\n### ", cleaned)
    cleaned = re.sub(r"(?i)([^\n])Tip(?=[A-Z])", r"\1\n\n### Tip\n", cleaned)
    cleaned = re.sub(r"(?im)^Tip(?=\s*[A-Z])", "### Tip\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def repair_research_reply(
    text: str,
    facts: ResearchFacts,
    *,
    packet: dict[str, Any] | None = None,
) -> str:
    cleaned = strip_advice_sentences(text)
    cleaned = strip_brochure_voice(cleaned)
    cleaned = strip_tool_name_leaks(cleaned)
    cleaned = strip_unknown_number_sentences(cleaned, facts)
    if packet is not None:
        cleaned = strip_ungrounded_catalyst_sentences(cleaned, packet)
    cleaned = _PLACEHOLDER_RE.sub("", cleaned)
    cleaned = polish_research_markdown(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return ensure_next_chips(cleaned)


def _deterministic_confidence(packet: dict[str, Any]) -> str:
    overview = get_run_overview(packet)
    meta = explain_confidence(packet)
    drivers = "\n".join(f"- {d}" for d in (meta.get("drivers") or []))
    return (
        f"### Why confidence is {meta.get('confidence')}\n"
        f"{meta.get('meaning')}\n\n"
        f"### Drivers on this run\n"
        f"{drivers}\n"
        f"- Model source: {meta.get('model_source')}\n"
        f"- Agreement across specialists: {meta.get('agreement')}\n"
        f"- Stance {overview.get('stance')} follows Performance trend "
        f"{overview.get('performance_trend')}.\n\n"
        f"### Tip\n"
        f"MIXED or conflicting news can make the picture messier to read, "
        f"but it does not by itself set Medium confidence.\n\n"
        f"{_DEFAULT_NEXT}"
    )


def _deterministic_focused_metric(packet: dict[str, Any], metric: str) -> str:
    overview = get_run_overview(packet)
    path = get_forecast_path(packet)
    ticker = overview.get("ticker")
    move = overview.get("projected_move") or "—"
    conf = explain_confidence(packet)

    if metric == "news":
        label = overview.get("news_sentiment")
        return (
            f"### What MIXED (and news labels) mean\n"
            f"News sentiment is the headline tone captured for this packet "
            f"(POSITIVE / NEGATIVE / MIXED / UNAVAILABLE).\n\n"
            f"### On {ticker} right now\n"
            f"- News is {label}\n"
            f"- MIXED means headlines pull both ways — useful tension to read "
            f"alongside trend, not an automatic confidence setter\n\n"
            f"### Tip\n"
            f"Medium confidence is about forecast-path trust, not MIXED news alone.\n\n"
            f"{_DEFAULT_NEXT}"
        )
    if metric == "trend":
        return (
            f"### What trend means\n"
            f"Trend is the short-horizon direction implied by the Performance "
            f"forecast path (BULLISH / BEARISH / SIDEWAYS). Stance usually follows it.\n\n"
            f"### On {ticker} right now\n"
            f"- Trend {overview.get('performance_trend')}; stance {overview.get('stance')}\n\n"
            f"{_DEFAULT_NEXT}"
        )
    if metric == "projected_move":
        return (
            f"### What projected move means\n"
            f"Signed % from last close to the final forecast session "
            f"(horizon {path.get('horizon') or '—'}). Plus = up path; minus = down path. "
            f"It describes the path stretch — not a promised live-market return.\n\n"
            f"### On {ticker} right now\n"
            f"- Projected move {move}\n\n"
            f"{_DEFAULT_NEXT}"
        )
    if metric == "risk":
        risk = get_risk_metrics(packet)
        metrics = risk.get("metrics") or {}
        metric_lines = "\n".join(
            f"- {key}: {value}" for key, value in list(metrics.items())[:6]
        ) or "- No coded risk metrics in this packet."
        return (
            f"### What Risk means\n"
            f"Risk is the downside / uncertainty tier for this run "
            f"(CONTAINED / MODERATE / ELEVATED, etc.).\n\n"
            f"### On {ticker} right now\n"
            f"- Risk label: {overview.get('risk_level')}\n"
            f"{metric_lines}\n\n"
            f"{_DEFAULT_NEXT}"
        )
    if metric == "financials":
        return (
            f"### What financial health means\n"
            f"Fundamentals health from the snapshot "
            f"(STRONG / ADEQUATE / STRESSED / UNAVAILABLE).\n\n"
            f"### On {ticker} right now\n"
            f"- Financials {overview.get('financial_health')}\n\n"
            f"{_DEFAULT_NEXT}"
        )
    if metric == "confidence":
        return _deterministic_confidence(packet)
    return _deterministic_glossary(packet)


def _deterministic_glossary(packet: dict[str, Any]) -> str:
    overview = get_run_overview(packet)
    path = get_forecast_path(packet)
    risk = get_risk_metrics(packet)
    move = overview.get("projected_move") or "—"
    conf = explain_confidence(packet)
    risk_metrics = risk.get("metrics") or {}
    vol = risk_metrics.get("realized_vol") or risk_metrics.get("annualized_vol")
    dd = risk_metrics.get("max_drawdown") or risk_metrics.get("historical_drawdown")
    risk_bits = []
    if vol is not None:
        try:
            risk_bits.append(
                f"realized/annualized vol about {float(vol):.1%}"
                if abs(float(vol)) <= 2
                else f"vol about {float(vol)}"
            )
        except (TypeError, ValueError):
            risk_bits.append(f"vol {vol}")
    if dd is not None:
        try:
            risk_bits.append(
                f"historical drawdown about {float(dd):.2%}"
                if abs(float(dd)) <= 2
                else f"drawdown {dd}"
            )
        except (TypeError, ValueError):
            risk_bits.append(f"drawdown {dd}")
    risk_extra = f" ({'; '.join(risk_bits)})" if risk_bits else ""

    return (
        f"### Labels on this run ({overview.get('ticker')})\n"
        f"- Trend {overview.get('performance_trend')}: direction implied by the short "
        f"forecast path (Performance). Stance usually follows this.\n"
        f"- Projected move {move}: signed % from last close to the final forecast "
        f"session (horizon {path.get('horizon') or '—'}). Plus = up path; minus = down path. "
        f"Not a guarantee of the live market move.\n"
        f"- News {overview.get('news_sentiment')}: headline tone for this packet "
        f"(POSITIVE / NEGATIVE / MIXED / UNAVAILABLE).\n"
        f"- Financials {overview.get('financial_health')}: fundamentals health from "
        f"the snapshot (STRONG / ADEQUATE / STRESSED / UNAVAILABLE).\n"
        f"- Risk {overview.get('risk_level')}: downside / uncertainty tier for this run"
        f"{risk_extra}.\n"
        f"- Confidence {overview.get('confidence')}: how much to trust the stance given "
        f"forecast model trust — not 'MIXED news alone.'\n\n"
        f"### Confidence on this run\n"
        f"{conf.get('meaning')}\n"
        + "\n".join(f"- {d}" for d in (conf.get("drivers") or []))
        + f"\n\n### Tip\n"
        f"Specialists can disagree. That tension is useful research signal, not an error.\n\n"
        f"{_DEFAULT_NEXT}"
    )


def _deterministic_stance_why(packet: dict[str, Any]) -> str:
    """Why bullish/bearish — labels only; no free-form catalyst invention."""
    overview = get_run_overview(packet)
    comparison = compare_labels(packet)
    move = overview.get("projected_move") or "—"
    ticker = overview.get("ticker")
    stance = overview.get("stance")
    rows = comparison.get("rows") or []
    bullets = "\n".join(
        f"- {row['dimension']}: {row['label']} ({row['lean']})" for row in rows
    )
    return (
        f"### Why {ticker} is {stance}\n"
        f"On this desk, stance follows the Performance trend.\n\n"
        f"### Drivers in this packet\n"
        f"- Trend {overview.get('performance_trend')} → stance {stance}\n"
        f"- Projected move {move}\n"
        f"- Financials {overview.get('financial_health')}; "
        f"news {overview.get('news_sentiment')}; risk {overview.get('risk_level')}\n"
        f"- Confidence {overview.get('confidence')}\n\n"
        f"### Specialist labels\n"
        f"{bullets}\n\n"
        f"### Tip\n"
        f"Agreement signal: {comparison.get('agreement')}. "
        f"Ask for an agent note if you want the written specialist prose — "
        f"Research will not invent partnerships or catalysts outside the packet.\n\n"
        f"{_DEFAULT_NEXT}"
    )


def multi_packet_compare_moves_reply(
    desk: DeskContext,
    tickers: list[str],
    *,
    missing: list[str] | None = None,
) -> str:
    lines: list[str] = []
    for sym in tickers:
        pkt = desk.get(sym)
        if not pkt:
            continue
        overview = get_run_overview(pkt)
        path = get_forecast_path(pkt)
        lines.append(
            f"- {sym}: projected move {overview.get('projected_move') or '—'} "
            f"(trend {overview.get('performance_trend')}, "
            f"stance {overview.get('stance')}, "
            f"horizon {path.get('horizon') or '—'})"
        )
    missing = [m for m in (missing or []) if m]
    miss_block = ""
    if missing:
        miss_block = (
            f"\n### Not on this desk\n"
            + "\n".join(f"- {m}: no finished packet in this visit" for m in missing)
            + "\n"
        )
    return (
        f"### Projected moves on this desk\n"
        f"{chr(10).join(lines) or '- No matching desk packets.'}\n\n"
        f"### How to read projected move\n"
        f"- Signed % from last close to the final forecast session\n"
        f"- Plus = up path; minus = down path\n"
        f"- Describes the path stretch — not a promised live-market return\n"
        f"- Compare across tickers as research framing, not a ranking to trade\n"
        f"{miss_block}\n"
        f"{_DEFAULT_NEXT}"
    )


def multi_packet_risk_reply(
    desk: DeskContext,
    tickers: list[str],
    *,
    missing: list[str] | None = None,
) -> str:
    sections: list[str] = []
    for sym in tickers:
        pkt = desk.get(sym)
        if not pkt:
            continue
        overview = get_run_overview(pkt)
        risk = get_risk_metrics(pkt)
        metrics = risk.get("metrics") or {}
        metric_lines = "\n".join(
            f"  - {key}: {value}" for key, value in list(metrics.items())[:6]
        ) or "  - No coded risk metrics in this packet."
        sections.append(
            f"### {sym} — Risk {overview.get('risk_level')}\n"
            f"- Trend {overview.get('performance_trend')}; "
            f"news {overview.get('news_sentiment')}; "
            f"financials {overview.get('financial_health')}\n"
            f"- Stance {overview.get('stance')}; "
            f"projected move {overview.get('projected_move') or '—'}\n"
            f"{metric_lines}"
        )
    missing = [m for m in (missing or []) if m]
    miss_block = ""
    if missing:
        miss_block = (
            "\n### Not on this desk\n"
            + "\n".join(f"- {m}: no finished packet in this visit" for m in missing)
            + "\n"
        )
    return (
        f"### Risk across named desk runs\n"
        f"Figures below come only from finished packets on this desk.\n\n"
        + "\n\n".join(sections)
        + f"\n{miss_block}\n"
        f"### Caveat\n"
        f"Research support only — not trade advice.\n\n"
        f"{_DEFAULT_NEXT}"
    )


def deterministic_research_answer(packet: dict[str, Any], *, question: str = "") -> str:
    """Safe fallback grounded only in packet tools (single packet)."""
    overview = get_run_overview(packet)
    comparison = compare_labels(packet)
    move = overview.get("projected_move") or "—"
    rows = comparison.get("rows") or []
    bullets = "\n".join(
        f"- {row['dimension']}: {row['label']} ({row['lean']})" for row in rows
    )
    q = (question or "").lower()

    if is_open_run_meta_question(question):
        return open_run_meta_reply(packet)

    if re.search(r"\bwhy\s+(?:is|was)\b", q) and re.search(
        r"\b(bullish|bearish|neutral)\b", q
    ):
        return _deterministic_stance_why(packet)
    if any(k in q for k in ("why bullish", "why bearish", "why neutral", "why is it bullish")):
        return _deterministic_stance_why(packet)

    if any(w in q for w in ("confidence", "why medium", "why low", "why high", "how sure")):
        return _deterministic_confidence(packet)

    focused = focused_metric_key(question)
    if focused:
        return _deterministic_focused_metric(packet, focused)

    if prefers_deterministic_answer(question) or any(
        w in q for w in ("what does", "what do these", "mean when", "metric", "labels")
    ):
        return _deterministic_glossary(packet)

    title = "Run snapshot"
    if "disagree" in q or "conflict" in q:
        title = "Where the specialists disagree"
    elif "risk" in q:
        title = "Risk on this run"
        risk = get_risk_metrics(packet)
        metrics = risk.get("metrics") or {}
        metric_lines = "\n".join(
            f"- {key}: {value}" for key, value in list(metrics.items())[:8]
        ) or "- No coded risk metrics in this packet."
        return (
            f"### {title}\n"
            f"Risk label on this run: {overview.get('risk_level')}.\n\n"
            f"### Coded metrics\n"
            f"{metric_lines}\n\n"
            f"### Context\n"
            f"- Stance {overview.get('stance')} / trend {overview.get('performance_trend')}\n"
            f"- News {overview.get('news_sentiment')}; financials {overview.get('financial_health')}\n"
            f"- Projected move {move}\n\n"
            f"### Caveat\n"
            f"Research support only — not trade advice.\n\n"
            f"{_DEFAULT_NEXT}"
        )

    return (
        f"### {title}\n"
        f"{overview.get('ticker')} — stance {overview.get('stance')}, "
        f"confidence {overview.get('confidence')}, projected move {move}.\n\n"
        f"### Labels\n"
        f"{bullets}\n\n"
        f"### Reading tip\n"
        f"{comparison.get('note')}\n"
        f"Agreement signal: {comparison.get('agreement')}.\n\n"
        f"### Caveat\n"
        f"Research support only — not trade advice. Figures come from this run’s packet.\n\n"
        f"{_DEFAULT_NEXT}"
    )


def build_research_system_prompt(
    facts: ResearchFacts,
    *,
    desk_tickers: list[str] | None = None,
    active_ticker: str | None = None,
) -> str:
    move = format_projected_move(facts.projected_move_pct)
    desk = desk_tickers or [facts.ticker]
    active = str(active_ticker or facts.ticker).upper()
    desk_line = ", ".join(desk)
    return f"""You are Research Co-Pilot for Stock Intelligence.

Answer from the PACKET FACTS in the user message (one or more finished desk runs).
Desk visit packets: {desk_line}. Active results view: {active}.
If the question names a desk ticker/company, use that packet. If none named, use the target packet facts provided.
Voice: Google-product support tone — scannable, concrete, no fluff.

OUTPUT FORMAT (strict):
### Title
One clear sentence.

### Section
- Short bullet
- Short bullet

Use 2–4 sections. Always end with:
{_DEFAULT_NEXT}

Hard rules:
- ### headings and - bullets only. No bold/italics (** __ *).
- No self-intro ("I am Research Co-Pilot…"). No "Analysis Scope" / "Analysis Purpose" essays.
- Do not tell the user to click This session when the packet facts are already provided.
- Cite labels and numbers exactly from PACKET FACTS.
- Never invent prices, headlines, URLs, metrics, partnerships, deals, or catalysts.
- Only mention a partnership/deal/year if that exact claim already appears in PACKET FACTS notes.
- Never name tools. Never write [packet figure].
- No buy/sell advice. Trade questions → reply exactly:
{OUT_OF_SCOPE}

Metric meanings on this desk:
- Trend: short-horizon forecast path direction (Performance). Stance follows trend.
- Projected move: % from last close to final forecast session (+ up path, − down path).
- News: headline tone for this packet (can be MIXED or UNAVAILABLE).
- Financials: fundamentals health from the snapshot.
- Risk: downside / uncertainty tier from coded metrics on this run.
- Confidence: forecast-trust rating (Low often = persistence/repaired path; Medium = normal model path). 
  Do NOT say Medium exists only because news is MIXED. Disagreement is a reading tip, not the sole confidence driver.

LOCKED CONTEXT (primary target):
- Ticker: {facts.ticker}
- Trend: {facts.performance_trend}
- News: {facts.news_sentiment}
- Financials: {facts.financial_health}
- Risk: {facts.risk_level}
- Stance: {facts.stance}
- Confidence: {facts.confidence}
- Projected move: {move}
"""
