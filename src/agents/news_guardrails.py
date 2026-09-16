"""Harness for the Market Expert (agent 2): news facts, prompt, validate, repair.

Pure logic stays LLM-free so CI can eval fixtures without Ollama or live news APIs.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from logger.context import set_ticker
from logger.logger import get_logger, log_event
from src.agents.llm import message_text

logger = get_logger("sip.agent2.harness")

SENTIMENTS = ("POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL", "UNAVAILABLE")
MIN_ANALYSIS_CHARS = 180
_SENTIMENT_LINE_RE = re.compile(
    r"(?im)^\s*sentiment\s*:\s*(positive|negative|mixed|neutral|unavailable)\b"
)
_ANALYSIS_BLOCK_RE = re.compile(
    r"(?is)^\s*analysis\s*:\s*(.+?)(?=^\s*(?:headlines|implications|themes|drivers|caveats?|key points)\s*:|\Z)",
    re.MULTILINE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class NewsFacts:
    ticker: str
    status: str
    provider: str | None
    headlines: tuple[str, ...]
    allowed_sentiment: str
    article_blob: str


@dataclass(frozen=True)
class NewsGuardrailResult:
    ok: bool
    parsed_sentiment: str | None
    errors: tuple[str, ...]


def _normalize_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall((text or "").lower()) if len(token) > 2}


def extract_headlines(news: dict[str, Any] | None) -> list[str]:
    if not news:
        return []
    headlines: list[str] = []
    for article in news.get("articles") or []:
        if not isinstance(article, dict):
            continue
        headline = str(article.get("headline") or "").strip()
        if headline:
            headlines.append(headline)
    return headlines


def build_news_facts(news: dict[str, Any] | None, *, ticker: str = "") -> NewsFacts:
    """Derive deterministic facts from the news tool payload."""
    payload = news or {}
    symbol = str(payload.get("ticker") or ticker or "").upper()
    status = str(payload.get("status") or "error")
    headlines = tuple(extract_headlines(payload))
    provider = payload.get("provider")
    provider_name = str(provider) if provider else None 

    parts: list[str] = []
    for article in payload.get("articles") or []:
        if not isinstance(article, dict):
            continue
        parts.extend(
            [
                str(article.get("headline") or ""),
                str(article.get("summary") or ""),
            ]
        )
    article_blob = "\n".join(parts)

    if status != "ok" or not headlines:
        allowed = "UNAVAILABLE"
    else:
        allowed = "HAS_ARTICLES"

    return NewsFacts(
        ticker=symbol,
        status=status,
        provider=provider_name,
        headlines=headlines,
        allowed_sentiment=allowed,
        article_blob=article_blob,
    )


def parse_sentiment_label(text: str) -> str | None:
    if not text:
        return None
    line = _SENTIMENT_LINE_RE.search(text)
    if line:
        return line.group(1).upper()
    upper = text.upper()
    for label in SENTIMENTS:
        if re.search(rf"\b{label}\b", upper):
            return label
    return None


def _analysis_body(text: str) -> str:
    match = _ANALYSIS_BLOCK_RE.search(text or "")
    if match:
        return match.group(1).strip()
    for line in (text or "").splitlines():
        if re.match(r"(?i)^\s*analysis\s*:", line):
            return re.sub(r"(?i)^\s*analysis\s*:", "", line).strip()
    return ""


def _headlines_section_bullets(text: str) -> list[str]:
    """Bullets under Headlines: (or legacy Drivers:/Themes:) — not Implications."""
    lines = (text or "").splitlines()
    bullets: list[str] = []
    in_headlines = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if re.match(r"(?i)^(headlines|drivers|themes)\s*:", stripped):
            in_headlines = True
            rest = re.sub(r"(?i)^(headlines|drivers|themes)\s*:", "", stripped).strip()
            if rest.startswith(("-", "*")):
                bullets.append(rest.lstrip("-* ").strip())
            continue
        if re.match(
            r"(?i)^(analysis|implications|caveats?|key points|sentiment)\s*:",
            stripped,
        ):
            in_headlines = False
            continue
        if in_headlines and (stripped.startswith("-") or stripped.startswith("*")):
            bullets.append(stripped.lstrip("-* ").strip())
    return bullets


def _bullet_lines_grounded(text: str, facts: NewsFacts, *, min_overlap: int = 2) -> bool:
    """Require Headlines bullets to share tokens with fetched articles."""
    if facts.allowed_sentiment == "UNAVAILABLE":
        return True
    allowed = _normalize_tokens(facts.article_blob)
    if not allowed:
        return False
    bullets = _headlines_section_bullets(text)
    if not bullets:
        body = _SENTIMENT_LINE_RE.sub("", text or "")
        return len(_normalize_tokens(body) & allowed) >= min_overlap
    return len(_normalize_tokens("\n".join(bullets)) & allowed) >= min_overlap


def validate_news_analysis(text: str, facts: NewsFacts) -> NewsGuardrailResult:
    """Hard checks: sentiment label, grounded headlines, substantial Analysis."""
    errors: list[str] = []
    parsed = parse_sentiment_label(text)
    if parsed is None:
        errors.append("missing_sentiment_label")
    elif parsed not in SENTIMENTS:
        errors.append(f"invalid_sentiment:{parsed}")
    elif facts.allowed_sentiment == "UNAVAILABLE" and parsed != "UNAVAILABLE":
        errors.append(f"sentiment_mismatch:got={parsed}:expected=UNAVAILABLE")
    elif facts.allowed_sentiment == "HAS_ARTICLES" and parsed == "UNAVAILABLE":
        errors.append("sentiment_unavailable_despite_articles")

    if facts.allowed_sentiment == "HAS_ARTICLES":
        if not re.search(r"(?im)^\s*analysis\s*:", text or ""):
            errors.append("missing_analysis_section")
        else:
            body = _analysis_body(text)
            if len(body) < MIN_ANALYSIS_CHARS:
                errors.append(f"analysis_too_short:{len(body)}<{MIN_ANALYSIS_CHARS}")
        if not _bullet_lines_grounded(text, facts):
            errors.append("ungrounded_drivers")

        source_tokens = _normalize_tokens(facts.article_blob)
        for cleaned in _headlines_section_bullets(text):
            if len(cleaned) < 40:
                continue
            line_tokens = _normalize_tokens(cleaned)
            if len(line_tokens) >= 6 and len(line_tokens & source_tokens) == 0:
                errors.append("invented_headline_line")
                break

    return NewsGuardrailResult(ok=not errors, parsed_sentiment=parsed, errors=tuple(errors))


def deterministic_news_analysis(facts: NewsFacts) -> str:
    """Code fallback when the LLM fails guardrails after retry."""
    if facts.allowed_sentiment == "UNAVAILABLE":
        return (
            "Sentiment: UNAVAILABLE\n"
            "Headlines:\n"
            "- No usable ticker-relevant articles in the fetch window.\n"
            "Analysis:\n"
            "No reliable ticker-relevant headlines were available, so a news-based "
            "view cannot be formed without inventing coverage. Treat news as missing "
            "for this run and rely on the forecast path plus primary sources you trust.\n"
            "Implications:\n"
            "- Do not invent competitor or company events to fill the gap.\n"
            "Caveats: Treat news as missing; do not invent stories."
        )

    headline_lines = "\n".join(f"- {h}" for h in facts.headlines[:4]) or "- See fetched headlines"
    joined = "; ".join(facts.headlines[:3])
    return (
        "Sentiment: MIXED\n"
        f"Headlines:\n{headline_lines}\n"
        "Analysis:\n"
        "Headlines were available but a full model write-up could not be validated. "
        f"Coverage includes: {joined}. Treat tone as provisional and open the source "
        "links; peer items are context only, not company confirmation.\n"
        "Implications:\n"
        "- Verify each headline against the linked article before acting on it.\n"
        "- Thin or mixed coverage usually means waiting for clearer confirmation.\n"
        "Caveats: Fallback after guardrail failure; coverage may be thin or stale."
    )


def repair_news_analysis(text: str, facts: NewsFacts) -> str:
    """Force a valid Sentiment line; fall back to deterministic text if still invalid."""
    body = _SENTIMENT_LINE_RE.sub("", (text or "").strip()).strip()
    if facts.allowed_sentiment == "UNAVAILABLE":
        return deterministic_news_analysis(facts)
    if not body:
        return deterministic_news_analysis(facts)
    if not re.search(r"(?im)^\s*headlines\s*:", body):
        bullets = "\n".join(f"- {h}" for h in facts.headlines[:6])
        body = f"Headlines:\n{bullets}\n{body}"
    if not re.search(r"(?im)^\s*analysis\s*:", body):
        preview = "; ".join(facts.headlines[:3]) or "see fetched headlines"
        body = (
            f"{body}\nAnalysis:\nCoverage includes {preview}. "
            + deterministic_news_analysis(facts).split("Analysis:\n", 1)[-1].split("Implications:", 1)[0]
        )
    if not re.search(r"(?im)^\s*caveat", body):
        body = f"{body}\nCaveats: News can be incomplete or delayed."
    sentiment = parse_sentiment_label(text) or "MIXED"
    if sentiment == "UNAVAILABLE":
        sentiment = "MIXED"
    repaired = f"Sentiment: {sentiment}\n{body}".strip()
    if validate_news_analysis(repaired, facts).ok:
        return repaired
    return deterministic_news_analysis(facts)


def build_news_prompt(ticker: str, news_raw: str, facts: NewsFacts) -> str:
    if facts.allowed_sentiment == "UNAVAILABLE":
        return f"""You are a market strategist for {ticker}.

NEWS:
{news_raw}

There are no usable headlines. Write EXACTLY:

Sentiment: UNAVAILABLE
Headlines:
- No usable articles in the fetch window.
Analysis:
No reliable ticker-relevant headlines were fetched, so a news-based view cannot be formed without inventing coverage. Explain briefly why inventing stories would be harmful and that the user should treat news as missing.
Implications:
- Do not invent competitor or company events.
Caveats: Do not invent news.

Rules:
- Do not invent headlines or companies.
- Do not discuss price forecasts.
"""

    bullets = "\n".join(f"- {h}" for h in facts.headlines[:5])
    return f"""You are a market strategist. Write a compact news briefing for {ticker} — clear and useful, not an essay.

NEWS (ticker + any RELATED/COMPETITOR items):
{news_raw}

Priority headlines:
{bullets}

Use EXACTLY this structure:

Sentiment: POSITIVE|NEGATIVE|MIXED|NEUTRAL
Headlines:
- <3–5 important bullets, close to source wording>
Analysis:
<1–2 short paragraphs (about 80–140 words). Synthesize what the tape implies for {ticker}; note conflicts; mention peer/competitor items only if present in NEWS. Do not restate every headline.>
Implications:
- <crisp takeaway>
- <crisp takeaway>
- <optional third takeaway>
Caveats:
<1–2 sentences on coverage gaps / uncertainty>

Rules:
- Medium length only — no long essays.
- Vary wording from run to run; keep the same facts and Sentiment label.
- Use ONLY information in NEWS; do not invent events.
- Conflicting headlines → MIXED.
- Do not discuss price forecasts.
"""


def run_news_harness(
    *,
    ticker: str,
    news: dict[str, Any] | None,
    news_raw: str,
    llm: Any,
    invoke: Callable[..., Any] | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Prompt → validate → retry → repair/fallback. Returns summary + metadata."""
    set_ticker(ticker)
    facts = build_news_facts(news, ticker=ticker)
    call = invoke or (lambda messages: llm.invoke(messages))
    attempts: list[dict[str, Any]] = []
    text = ""
    result = NewsGuardrailResult(ok=False, parsed_sentiment=None, errors=("not_run",))
    harness_started = time.perf_counter()

    log_event(
        logger,
        "agent2_harness_begin",
        step="agent2_begin",
        status="ok",
        data={
            "news_status": facts.status,
            "provider": facts.provider,
            "n_headlines": len(facts.headlines),
            "allowed_sentiment": facts.allowed_sentiment,
        },
    )

    for attempt in range(1, max_attempts + 1):
        prompt = build_news_prompt(ticker, news_raw, facts)
        if attempt > 1:
            prompt += (
                "\n\nPREVIOUS OUTPUT FAILED GUARDRAILS. "
                f"Errors: {', '.join(result.errors)}. "
                "Respond again with a medium-length Analysis using only the NEWS block."
            )
            log_event(
                logger,
                "agent2_retry",
                step="agent2_retry",
                status="retry",
                metrics={"attempt": attempt},
                data={"prior_errors": list(result.errors)},
            )
        else:
            log_event(
                logger,
                "agent2_llm_invoke",
                step="agent2_llm",
                status="started",
                metrics={"attempt": attempt},
            )

        invoke_started = time.perf_counter()
        response = call(
            [
                SystemMessage(
                    content=(
                        "You write compact market news briefings. "
                        "Stay medium length — useful, not long essays."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        invoke_ms = (time.perf_counter() - invoke_started) * 1000
        text = message_text(response)
        result = validate_news_analysis(text, facts)
        attempts.append(
            {
                "attempt": attempt,
                "ok": result.ok,
                "errors": list(result.errors),
                "parsed_sentiment": result.parsed_sentiment,
                "chars": len(text or ""),
                "duration_ms": round(invoke_ms, 2),
            }
        )
        log_event(
            logger,
            "agent2_validation",
            step="agent2_validation",
            status="success" if result.ok else "failed",
            duration_ms=invoke_ms,
            metrics={
                "attempt": attempt,
                "chars": len(text or ""),
                "ok": result.ok,
            },
            data={
                "sentiment": result.parsed_sentiment,
                "errors": list(result.errors),
            },
        )
        if result.ok:
            break

    repaired = False
    if not result.ok:
        log_event(
            logger,
            "agent2_repair",
            level=logging.WARNING,
            step="agent2_repair",
            status="repair",
            metrics={"after_attempts": max_attempts},
        )
        text = repair_news_analysis(text, facts)
        result = validate_news_analysis(text, facts)
        repaired = True
        if not result.ok:
            log_event(
                logger,
                "agent2_deterministic_fallback",
                level=logging.WARNING,
                step="agent2_fallback",
                status="fallback",
            )
            text = deterministic_news_analysis(facts)
            result = validate_news_analysis(text, facts)

    total_ms = (time.perf_counter() - harness_started) * 1000
    log_event(
        logger,
        "agent2_harness_end",
        step="agent2_end",
        status="success" if result.ok else "failed",
        duration_ms=total_ms,
        metrics={
            "repaired": repaired,
            "chars": len(text or ""),
            "attempts": len(attempts),
        },
        data={
            "sentiment": result.parsed_sentiment,
            "guardrail_ok": result.ok,
        },
    )

    return {
        "news_summary": text,
        "news_sentiment": result.parsed_sentiment or (
            "UNAVAILABLE" if facts.allowed_sentiment == "UNAVAILABLE" else "MIXED"
        ),
        "news_guardrail_ok": result.ok,
        "news_guardrail_errors": list(result.errors),
        "news_repaired": repaired,
        "news_attempts": attempts,
        "news_duration_ms": round(total_ms, 2),
        "news_facts": {
            "status": facts.status,
            "provider": facts.provider,
            "n_headlines": len(facts.headlines),
            "allowed_sentiment": facts.allowed_sentiment,
        },
    }
