"""Harness for the Market Expert (agent 2): news facts, prompt, validate, repair.

Pure logic stays LLM-free so CI can eval fixtures without Ollama or live news APIs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import SystemMessage

from logger.logger import get_logger
from src.agents.llm import message_text

logger = get_logger()

SENTIMENTS = ("POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL", "UNAVAILABLE")
_SENTIMENT_LINE_RE = re.compile(
    r"(?im)^\s*sentiment\s*:\s*(positive|negative|mixed|neutral|unavailable)\b"
)
# Normalize for overlap checks: letters/digits only, lowercased tokens.
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
        # LLM must choose among these; we do not auto-pick POSITIVE/NEGATIVE from keywords
        # in v1 — only enforce UNAVAILABLE vs "some real articles exist".
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


def _drivers_grounded(text: str, facts: NewsFacts, *, min_overlap: int = 2) -> bool:
    """Require driver lines to share tokens with fetched article text."""
    if facts.allowed_sentiment == "UNAVAILABLE":
        return True
    allowed = _normalize_tokens(facts.article_blob)
    if not allowed:
        return False
    driver_lines = [
        line
        for line in (text or "").splitlines()
        if line.strip().lower().startswith("drivers:")
        or line.strip().startswith("-")
        or line.strip().startswith("*")
    ]
    if not driver_lines:
        # Fall back to whole body minus the Sentiment line.
        body = _SENTIMENT_LINE_RE.sub("", text or "")
        overlap = len(_normalize_tokens(body) & allowed)
        return overlap >= min_overlap
    overlap = len(_normalize_tokens("\n".join(driver_lines)) & allowed)
    return overlap >= min_overlap


def validate_news_analysis(text: str, facts: NewsFacts) -> NewsGuardrailResult:
    """Hard checks: sentiment label + no invented headline material."""
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

    if facts.allowed_sentiment == "HAS_ARTICLES" and not _drivers_grounded(text, facts):
        errors.append("ungrounded_drivers")

    # Reject a whole headline-looking line that shares almost no tokens with sources.
    if facts.headlines and parsed and parsed != "UNAVAILABLE":
        source_tokens = _normalize_tokens(facts.article_blob)
        for line in (text or "").splitlines():
            cleaned = line.strip()
            if len(cleaned) < 40:
                continue
            if cleaned.lower().startswith(("sentiment:", "caveat:", "drivers:")):
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
            "Drivers: No reliable headlines were fetched for "
            f"{facts.ticker or 'this ticker'}.\n"
            "Caveat: Treat news coverage as missing; do not invent stories."
        )
    joined = "; ".join(facts.headlines[:3])
    return (
        "Sentiment: MIXED\n"
        f"Drivers: - Coverage includes: {joined}\n"
        "Caveat: Fallback summary used after guardrail failure; verify sources."
    )


def repair_news_analysis(text: str, facts: NewsFacts) -> str:
    """Force a valid Sentiment line; fall back to deterministic text if still invalid."""
    body = _SENTIMENT_LINE_RE.sub("", (text or "").strip()).strip()
    if facts.allowed_sentiment == "UNAVAILABLE":
        return deterministic_news_analysis(facts)
    if not body:
        return deterministic_news_analysis(facts)
    if not re.search(r"(?im)^\s*drivers\s*:", body):
        preview = "; ".join(facts.headlines[:2]) or "see fetched headlines"
        body = f"Drivers: - {preview}\n{body}"
    if not re.search(r"(?im)^\s*caveat\s*:", body):
        body = f"{body}\nCaveat: News can be incomplete or delayed."
    # Default repaired label when articles exist but model omitted/broke sentiment.
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
Drivers: No reliable headlines were fetched.
Caveat: Do not invent news.

Rules:
- Do not invent headlines or companies.
- Do not discuss price forecasts.
"""

    return f"""You are a market strategist summarizing news sentiment for {ticker}.

NEWS:
{news_raw}

Write EXACTLY this structure:
Sentiment: POSITIVE|NEGATIVE|MIXED|NEUTRAL
Drivers: - <bullet grounded in the NEWS headlines/summaries above>
- <optional second bullet>
Caveat: <one sentence about coverage quality, staleness, or uncertainty>

Rules:
- Use ONLY information present in NEWS.
- Articles were pre-filtered for this ticker; still do not invent events.
- Do not invent headlines, numbers, or events.
- If headlines conflict, use MIXED.
- Do not discuss LSTM forecasts or price targets.
- Do not add Market Stance or Confidence lines.
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
    facts = build_news_facts(news, ticker=ticker)
    call = invoke or (lambda messages: llm.invoke(messages))
    attempts: list[dict[str, Any]] = []
    text = ""
    result = NewsGuardrailResult(ok=False, parsed_sentiment=None, errors=("not_run",))

    logger.info(
        "[agent2.harness] begin ticker=%s status=%s provider=%s headlines=%s allowed=%s",
        ticker,
        facts.status,
        facts.provider,
        len(facts.headlines),
        facts.allowed_sentiment,
    )

    for attempt in range(1, max_attempts + 1):
        prompt = build_news_prompt(ticker, news_raw, facts)
        if attempt > 1:
            prompt += (
                "\n\nPREVIOUS OUTPUT FAILED GUARDRAILS. "
                f"Errors: {', '.join(result.errors)}. "
                "Respond again using only the NEWS block."
            )
            logger.info(
                "[agent2.harness] retry ticker=%s attempt=%s prior_errors=%s",
                ticker,
                attempt,
                list(result.errors),
            )
        else:
            logger.info("[agent2.harness] llm_invoke ticker=%s attempt=%s", ticker, attempt)
        response = call([SystemMessage(content=prompt)])
        text = message_text(response)
        result = validate_news_analysis(text, facts)
        attempts.append(
            {
                "attempt": attempt,
                "ok": result.ok,
                "errors": list(result.errors),
                "parsed_sentiment": result.parsed_sentiment,
            }
        )
        logger.info(
            "[agent2.harness] validate ticker=%s attempt=%s ok=%s sentiment=%s errors=%s",
            ticker,
            attempt,
            result.ok,
            result.parsed_sentiment,
            list(result.errors),
        )
        if result.ok:
            break

    repaired = False
    if not result.ok:
        logger.warning(
            "[agent2.harness] repair ticker=%s after_attempts=%s",
            ticker,
            max_attempts,
        )
        text = repair_news_analysis(text, facts)
        result = validate_news_analysis(text, facts)
        repaired = True
        if not result.ok:
            logger.warning("[agent2.harness] deterministic_fallback ticker=%s", ticker)
            text = deterministic_news_analysis(facts)
            result = validate_news_analysis(text, facts)

    logger.info(
        "[agent2.harness] end ticker=%s sentiment=%s ok=%s repaired=%s",
        ticker,
        result.parsed_sentiment,
        result.ok,
        repaired,
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
        "news_facts": {
            "status": facts.status,
            "provider": facts.provider,
            "n_headlines": len(facts.headlines),
            "allowed_sentiment": facts.allowed_sentiment,
        },
    }
