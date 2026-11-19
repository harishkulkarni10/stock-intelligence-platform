"""Live news path for Research Co-Pilot — works with an empty desk.

Flow:
1. Resolve ticker from the user question (symbol or company name).
2. Fetch headlines via get_news (same tool the Market expert uses).
3. Run the news harness (Market expert) for sentiment + specialist note.
4. Have Research LLM read that agent output + headlines and answer the user.
5. Validate: no invented URLs/headlines; no buy/sell advice.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from logger.logger import get_logger, log_event
from src.agents.llm import get_chat_llm, message_text
from src.agents.news_guardrails import run_news_harness
from src.agents.assistant_voice import (
    clean_assistant_markdown,
    format_live_news_reply,
    harness_to_chat_reply,
    strip_market_harness_preface,
)
from src.agents.research_guardrails import (
    ADVICE_META,
    OUT_OF_SCOPE,
    _DEFAULT_NEXT,
    ensure_next_chips,
    is_advice_meta_question,
    is_advice_question,
    polish_research_markdown,
    strip_brochure_voice,
    strip_tool_name_leaks,
)
from src.agents.tools import (
    NEWS_CHAT_DEFAULT_LIMIT,
    NEWS_DEFAULT_LOOKBACK_DAYS,
    format_news_for_prompt,
    get_news,
)
from src.data.ingestion import normalize_ticker

logger = get_logger("sip.research_live_news")

# Second LLM (chat polish) — skip when slow; fall back to harness voice.
LIVE_NEWS_SYNTH_TIMEOUT_SEC = float(os.getenv("LIVE_NEWS_SYNTH_TIMEOUT_SEC", "45"))

_NEWS_HINT_RE = re.compile(
    r"(?i)\b("
    r"news|headline|headlines|article|articles|press|coverage|"
    r"sentiment|what(?:'s| is) (?:going on|happening)|"
    r"latest (?:on|about|hot)|recent (?:news|headlines|coverage)|"
    r"hot news|past week|this morning|today and for"
    r")\b"
)
_CLAIM_OR_EVENT_RE = re.compile(
    r"(?i)\b("
    r"bankrupt|bankruptcy|delist|delisting|"
    r"market cap|wiped|halved|cut in half|"
    r"fired|firings|ceo\b|chief executive|"
    r"jensen huang|according to (?:the )?news|"
    r"true or false|is it true"
    r")\b"
)
_LIVE_NEXT = (
    "NEXT: Latest headlines for {ticker}|Explain disagreement|Walk through Risk|"
    "Why this confidence?|Forecast path"
)
# Simple “give me headlines” asks — skip the second Research LLM after harness.
_SIMPLE_HEADLINES_ASK_RE = re.compile(
    r"(?i)^\s*(?:"
    r"(?:what(?:'s| is)\s+(?:the\s+)?)?(?:latest|recent)\s+(?:news|headlines|coverage)\b|"
    r"(?:latest|recent)\s+headlines?\b|"
    r"(?:news|headlines)\s+(?:for|on|about|today)\b|"
    r"(?:tell me|give me|show me)\s+(?:the\s+)?(?:latest\s+)?(?:news|headlines)\b|"
    r"[A-Za-z][A-Za-z0-9.&'\- ]{0,40}\s+news(?:\s+today)?\b"
    r").*$"
)
_NEEDS_SYNTH_RE = re.compile(
    r"(?i)\b("
    r"wiped|bankrupt|delist|halved|cut in half|market cap|"
    r"true|false|did |was |is nvidia|according to|"
    r"should i|buy|sell|why is|confidence|explain"
    r")\b"
)
_TICKER_TOKEN_RE = re.compile(r"\b([A-Za-z]{1,5})\b")
_TICKER_STOP = frozenset(
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
        "THE",
        "AND",
        "FOR",
        "ANY",
        "ALL",
        "HOW",
        "WHY",
        "WHO",
        "WHAT",
        "WHEN",
        "NEWS",
        "LATEST",
        "ABOUT",
        "TELL",
        "GIVE",
        "SHOW",
        "NEED",
        "WANT",
        "FROM",
        "WITH",
        "THIS",
        "THAT",
        "HAVE",
        "HAS",
        "ARE",
        "WAS",
        "GET",
        "CAN",
        "YOU",
        "BOT",
        "ASK",
        "DID",
        "HAD",
        "ITS",
        "OUR",
        "OUT",
        "OWN",
        "PER",
        "SAY",
        "SEE",
        "SET",
        "TOO",
        "VIA",
        "YET",
        "HALF",
        "TRUE",
        "FAKE",
        "JUST",
        "ONLY",
        "OVER",
        "ONCE",
        "INTO",
        "LIKE",
        "THAN",
        "THEN",
        "ALSO",
        "BEEN",
        "BEING",
        "DOES",
        "DONE",
        "EACH",
        "EVEN",
        "EVER",
        "FROM",
        "HERE",
        "HIGH",
        "LAST",
        "LONG",
        "MANY",
        "MUCH",
        "NEAR",
        "NEXT",
        "SAME",
        "SOME",
        "SUCH",
        "SURE",
        "TAKE",
        "THAN",
        "THEM",
        "THEY",
        "THIS",
        "VERY",
        "WELL",
        "WERE",
        "WILL",
        "WITH",
        "YOUR",
    }
)

# Spoken / brand names → ticker (global, not desk-scoped).
_LIVE_BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple",),
    "MSFT": ("microsoft",),
    "GOOGL": ("google", "alphabet"),
    "GOOG": ("google", "alphabet"),
    "AMZN": ("amazon",),
    "META": ("facebook", "meta platforms", "meta"),
    "TSLA": ("tesla",),
    "NVDA": ("nvidia",),
    "NFLX": ("netflix",),
    "AMD": ("amd", "advanced micro devices"),
    "INTC": ("intel",),
    "ORCL": ("oracle",),
    "IBM": ("ibm",),
    "JPM": ("jp morgan", "jpmorgan", "j p morgan", "chase"),
    "GS": ("goldman", "goldman sachs"),
    "MS": ("morgan stanley",),
    "BAC": ("bank of america", "bofa"),
    "WMT": ("walmart", "wal mart"),
    "KO": ("coke", "coca cola", "coca-cola", "cocacola"),
    "PEP": ("pepsi", "pepsico"),
    "DIS": ("disney",),
    "BA": ("boeing",),
    "CAT": ("caterpillar",),
    "CSCO": ("cisco",),
    "CRM": ("salesforce",),
    "BLK": ("blackrock",),
    "PLTR": ("palantir",),
    "MU": ("micron",),
    "AVGO": ("broadcom",),
    "COST": ("costco",),
    "NKE": ("nike",),
    "V": ("visa",),
    "MA": ("mastercard",),
}

NEED_TICKER_NEWS = (
    "### Need a ticker\n"
    "Ask about news for a specific company or symbol "
    "(e.g. Latest headlines for TSLA, or Goldman Sachs news).\n\n"
    "### Tip\n"
    "No desk run is required for live news questions.\n\n"
    f"{_DEFAULT_NEXT}"
)

NEWS_UNAVAILABLE = (
    "### News unavailable\n"
    "Live headlines could not be fetched for this ticker right now.\n\n"
    "### Tip\n"
    "Try again in a moment, or run a full analyze on the desk for a packet-backed note.\n\n"
    f"{_DEFAULT_NEXT}"
)


def is_news_question(message: str) -> bool:
    return wants_live_news(message)


_DESK_PACKET_QA_RE = re.compile(
    r"(?i)\b("
    r"explain disagreement|walk through risk|why this confidence|"
    r"why is stance|why neutral|why bullish|why bearish|"
    r"forecast path|projected move|on this run|this run|"
    r"disagreement across|bull vs bear"
    r")\b"
)
_LIVE_REFRESH_RE = re.compile(
    r"(?i)\b("
    r"latest|recent|headlines?|hot news|past week|today|"
    r"this morning|live |according to (?:the )?news|coverage"
    r")\b"
)


def wants_live_news(message: str) -> bool:
    """Live headlines / claims / news events — prefer live fetch over desk packet."""
    text = message or ""
    if _DESK_PACKET_QA_RE.search(text) and not _LIVE_REFRESH_RE.search(text):
        return False
    if _NEWS_HINT_RE.search(text):
        return True
    if _CLAIM_OR_EVENT_RE.search(text):
        return True
    if re.search(r"(?i)\b(earnings|outlook beat|after.?hours)\b", text):
        return True
    return False


_FOLLOW_UP_NEWS_RE = re.compile(
    r"(?i)\b("
    r"ceo|jensen|huang|what happened|what's with|why did|you said|"
    r"clarify|explain that|tell me more|that story|interim|go on"
    r")\b"
)
_LIVE_THREAD_MARKERS = (
    "worth a look",
    "— live news",
    "latest from live",
    "**worth a look**",
    "headlines",
)

NEWS_REPLY_CACHE_TTL_SEC = int(os.getenv("NEWS_REPLY_CACHE_TTL_SEC", "180"))
LIVE_HISTORY_TURNS = 6


@dataclass
class _ReplyCacheEntry:
    reply: str
    payload: dict[str, Any]
    expires: float


_REPLY_CACHE: dict[str, _ReplyCacheEntry] = {}


def _cache_key(ticker: str, question: str) -> str:
    norm = re.sub(r"\s+", " ", (question or "").lower().strip())
    return hashlib.sha256(f"{ticker}|{norm}".encode()).hexdigest()[:32]


def _cache_get(key: str) -> _ReplyCacheEntry | None:
    entry = _REPLY_CACHE.get(key)
    if not entry or entry.expires < time.time():
        _REPLY_CACHE.pop(key, None)
        return None
    return entry


def _cache_put(key: str, reply: str, payload: dict[str, Any]) -> None:
    _REPLY_CACHE[key] = _ReplyCacheEntry(
        reply=reply,
        payload=payload,
        expires=time.time() + NEWS_REPLY_CACHE_TTL_SEC,
    )


def is_live_news_follow_up(
    message: str,
    history: list[dict[str, str]] | None,
) -> bool:
    if not history or not (message or "").strip():
        return False
    if wants_live_news(message):
        return False
    if not _FOLLOW_UP_NEWS_RE.search(message) and len(message.split()) > 14:
        return False
    for turn in reversed(history[-LIVE_HISTORY_TURNS:]):
        blob = (turn.get("content") or "").lower()
        if any(marker in blob for marker in _LIVE_THREAD_MARKERS):
            return True
    return False


def should_route_live_news(
    message: str,
    history: list[dict[str, str]] | None,
) -> bool:
    return wants_live_news(message) or is_live_news_follow_up(message, history)


def _prefer_fast_headlines_only(
    question: str,
    history: list[dict[str, str]] | None,
) -> bool:
    """One LLM (Market expert only) — harness voice, not a second synthesis call."""
    if not _is_simple_headlines_ask(question):
        return False
    if is_live_news_follow_up(question, history):
        return False
    for turn in history or []:
        if turn.get("role") == "assistant" and (turn.get("content") or "").strip():
            return False
    return True


def _format_history_for_prompt(history: list[dict[str, str]] | None) -> str:
    chunks: list[str] = []
    for turn in (history or [])[-LIVE_HISTORY_TURNS:]:
        role = (turn.get("role") or "user").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        if len(content) > 900:
            content = content[:899] + "…"
        chunks.append(f"{label}: {content}")
    return "\n\n".join(chunks)


def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def resolve_ticker_from_context(
    message: str,
    history: list[dict[str, str]] | None = None,
    *,
    active_ticker: str | None = None,
) -> str | None:
    """Ticker from question, then recent chat, then active desk view."""
    ticker = resolve_ticker_from_question(message)
    if ticker:
        return ticker
    if (
        active_ticker
        and (
            _CLAIM_OR_EVENT_RE.search(message or "")
            or _FOLLOW_UP_NEWS_RE.search(message or "")
        )
    ):
        return active_ticker.upper()
    for turn in reversed(history or []):
        role = (turn.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        ticker = resolve_ticker_from_question(turn.get("content") or "")
        if ticker:
            return ticker
    return None


def resolve_ticker_from_question(message: str) -> str | None:
    """Best-effort ticker from symbol or company/brand phrasing."""
    q = message or ""
    q_norm = _normalize_phrase(q)
    q_compact = q_norm.replace(" ", "")

    # Brand / company aliases (prefer longer aliases first via sorted length).
    scored: list[tuple[int, str]] = []
    for ticker, aliases in _LIVE_BRAND_ALIASES.items():
        for alias in aliases:
            an = _normalize_phrase(alias)
            if not an:
                continue
            if an in q_norm or an.replace(" ", "") in q_compact:
                scored.append((len(an), ticker))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    # Explicit symbol tokens.
    for match in _TICKER_TOKEN_RE.finditer(q):
        raw = match.group(1)
        token = raw.upper()
        if token in _TICKER_STOP or len(token) < 1:
            continue
        if token in _LIVE_BRAND_ALIASES:
            return token
        # Accept common equity tickers (2–5 letters) when capitalized or all-caps.
        if len(token) >= 2 and (raw.isupper() or raw[0].isupper()):
            try:
                return normalize_ticker(token)
            except Exception:  # noqa: BLE001
                return token
    return None


def _article_allowlist(news: dict[str, Any]) -> tuple[list[str], list[str]]:
    titles: list[str] = []
    urls: list[str] = []
    for item in news.get("articles") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("headline") or item.get("title") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        if title:
            titles.append(title)
        if url:
            urls.append(url)
    return titles, urls


def _live_next_chips(ticker: str) -> str:
    return _LIVE_NEXT.format(ticker=ticker)


def _news_fetch_plan(question: str) -> tuple[int, int]:
    q = (question or "").lower()
    limit = NEWS_CHAT_DEFAULT_LIMIT
    if re.search(r"(?i)\bpast week\b|1 week|seven day|last 7", q):
        limit = max(limit, 15)
    lookback = NEWS_DEFAULT_LOOKBACK_DAYS
    return limit, lookback


def _headline_titles(
    news: dict[str, Any],
    *,
    max_items: int = 10,
) -> list[str]:
    titles: list[str] = []
    for item in (news.get("articles") or []):
        if item.get("scope") == "peer":
            continue
        if len(titles) >= max_items:
            break
        title = str(item.get("headline") or item.get("title") or "").strip()
        if title:
            titles.append(title)
    return titles


def _format_headline_bullets(
    news: dict[str, Any],
    *,
    max_items: int = 10,
    question: str | None = None,
) -> tuple[str, str]:
    """Return (all bullets markdown, optional CEO highlight block)."""
    lines: list[str] = []
    ceo_lines: list[str] = []
    q = (question or "").lower()
    focus_ceo = any(w in q for w in ("ceo", "jensen", "huang", "leadership", "resign", "fired", "interim"))
    for item in (news.get("articles") or []):
        if item.get("scope") == "peer":
            continue
        if len(lines) >= max_items:
            break
        title = str(item.get("headline") or item.get("title") or "").strip()
        if not title:
            continue
        when = str(item.get("published") or item.get("date") or "").strip()[:10]
        prefix = f"{when} — " if when else ""
        line = f"- {prefix}{title}"
        lines.append(line)
        title_low = title.lower()
        if focus_ceo and any(k in title_low for k in ("ceo", "jensen", "huang", "interim", "leadership")):
            ceo_lines.append(line)
    bullets = "\n".join(lines) or "- No headlines kept."
    ceo_block = ""
    if ceo_lines:
        ceo_block = "**Leadership (from headlines)**\n" + "\n".join(ceo_lines[:5]) + "\n\n"
    return bullets, ceo_block


def _deterministic_news_reply(
    *,
    ticker: str,
    sentiment: str,
    note: str,
    news: dict[str, Any],
    compact: bool = False,
    question: str | None = None,
) -> str:
    del compact, question
    titles = _headline_titles(news, max_items=8)
    body = harness_to_chat_reply(
        ticker=ticker,
        harness_note=note,
        headline_titles=titles,
        sentiment=sentiment,
    )
    return clean_assistant_markdown(f"{body}\n\n{_live_next_chips(ticker)}")


def _is_simple_headlines_ask(question: str) -> bool:
    q = (question or "").strip()
    if not q or _NEEDS_SYNTH_RE.search(q):
        return False
    return bool(_SIMPLE_HEADLINES_ASK_RE.match(q))


def _validate_live_news_reply(
    reply: str,
    *,
    titles: list[str],
    urls: list[str],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    body = reply or ""
    if is_advice_question(body) or re.search(
        r"(?i)\b(buy|sell|accumulate|trim the position)\b", body
    ):
        # Soft: advice phrases in the answer body.
        if re.search(
            r"(?i)\b(should buy|should sell|buy now|sell now|go long|go short)\b",
            body,
        ):
            errors.append("advice_language")

    # URLs in the reply must be from the allowlist.
    for url in re.findall(r"https?://[^\s)\]>\"']+", body, flags=re.I):
        cleaned = url.rstrip(".,);")
        if cleaned not in urls and not any(
            u.startswith(cleaned) or cleaned.startswith(u) for u in urls
        ):
            errors.append(f"unknown_url:{cleaned[:80]}")

    # If the model invents a long quoted headline, require overlap with allowlist.
    for quoted in re.findall(r'"([^"]{12,160})"', body):
        q_low = quoted.lower()
        if not any(q_low[:40] in t.lower() or t.lower()[:40] in q_low for t in titles):
            # Only flag if it looks like a headline claim.
            if re.search(r"(?i)\b(reports?|announces?|says|said|headline)\b", body):
                errors.append("unknown_headline")
                break

    return (not errors, errors)


def _synthesize_from_agent(
    *,
    question: str,
    ticker: str,
    sentiment: str,
    agent_note: str,
    news: dict[str, Any],
    history: list[dict[str, str]] | None,
    llm,
) -> str:
    ticker_articles = [
        a
        for a in (news.get("articles") or [])
        if isinstance(a, dict) and a.get("scope") != "peer"
    ][:8]
    headline_lines = [
        f"- {(a.get('headline') or a.get('title') or '').strip()}"
        for a in ticker_articles
        if (a.get("headline") or a.get("title"))
    ]
    note_clip = strip_market_harness_preface(agent_note)
    if len(note_clip) > 900:
        note_clip = note_clip[:899] + "…"
    hist_blob = _format_history_for_prompt(history)
    system = f"""Desk assistant — concise chat answer about {ticker} from LIVE NEWS only.

STYLE:
- **{ticker}** then 2–3 short sentences (themes only).
- **Worth a look** — 4–6 bullets; label: one-line fact (e.g. "Berlin expansion: …").
- No provider names, timestamps, disclaimers, or stock phrases like "navigating a period of intense".
- Use CONVERSATION only when the user is following up.

RULES: Only facts in the expert note and headline list. No new numbers. No buy/sell.
End with:
{_live_next_chips(ticker)}
"""
    messages: list = [SystemMessage(content=system)]
    if hist_blob:
        messages.append(
            HumanMessage(content=f"CONVERSATION (this session):\n{hist_blob}")
        )
    user = (
        f"QUESTION: {question}\n\n"
        f"Sentiment: {sentiment}\n"
        f"Expert note:\n{note_clip}\n\n"
        f"Headlines:\n" + "\n".join(headline_lines) + "\n"
    )
    messages.append(HumanMessage(content=user))
    return message_text(llm.invoke(messages)).strip()


def _synthesize_with_timeout(**kwargs: Any) -> str | None:
    timeout = max(5.0, LIVE_NEWS_SYNTH_TIMEOUT_SEC)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_synthesize_from_agent, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            log_event(
                logger,
                "live_news_synth_timeout",
                level=logging.WARNING,
                step="research_live_news",
                status="timeout",
                data={
                    "ticker": kwargs.get("ticker"),
                    "timeout_sec": timeout,
                },
            )
            return None


def _reply_fails_headline_grounding(reply: str, titles: list[str]) -> bool:
    blob = " ".join(titles).lower()
    low = (reply or "").lower()
    if not blob:
        return False
    if ("jensen" in low or "huang" in low) and (
        "jensen" not in blob and "huang" not in blob
    ):
        return True
    if re.search(r"\$\s*\d", reply or "") and "$" not in blob and "billion" not in blob:
        if re.search(r"\d{2,}[^%]", reply or ""):
            return True
    if "fired" in low and "fire" not in blob and "fired" not in blob:
        return True
    if "bankrupt" in low and "bankrupt" not in blob:
        return True
    return False


def answer_live_news_question(
    message: str,
    history: list[dict[str, str]] | None = None,
    *,
    llm=None,
    news_fetcher=None,
    news_harness=None,
    active_ticker: str | None = None,
) -> dict[str, Any]:
    """Live news path — used even when a desk packet exists (fresh headlines)."""
    text = (message or "").strip()
    tools_used: list[str] = []

    if is_advice_meta_question(text):
        return {
            "reply": ADVICE_META,
            "out_of_scope": True,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
            "path": "advice_meta",
        }
    if is_advice_question(text):
        return {
            "reply": OUT_OF_SCOPE,
            "out_of_scope": True,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
            "path": "advice",
        }

    bare = text.strip().lower()
    if bare in ("news", "headlines", "coverage"):
        return {
            "reply": NEED_TICKER_NEWS,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
            "path": "need_ticker",
        }

    ticker = resolve_ticker_from_context(
        text, history, active_ticker=active_ticker
    )
    if not ticker:
        return {
            "reply": NEED_TICKER_NEWS,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
            "path": "need_ticker",
        }

    cache_key = _cache_key(ticker, text)
    cached = _cache_get(cache_key)
    if cached:
        hit = dict(cached.payload)
        hit["cache_hit"] = True
        return hit

    fetch = news_fetcher or get_news
    harness_fn = news_harness or run_news_harness
    client = llm if llm is not None else get_chat_llm()

    try:
        fetch_limit, lookback = _news_fetch_plan(text)
        news = fetch(
            ticker,
            limit=fetch_limit,
            lookback_days=lookback,
            focus_question=text,
            include_peers=False,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            "live_news_fetch_failed",
            level=logging.WARNING,
            step="research_live_news",
            status="error",
            data={"ticker": ticker, "error": str(exc)[:240]},
        )
        return {
            "reply": NEWS_UNAVAILABLE.replace("this ticker", ticker),
            "out_of_scope": False,
            "mode": "research",
            "tools_used": ["get_news"],
            "repaired": True,
            "path": "news_fetch_error",
        }

    tools_used.append("get_news")
    if not isinstance(news, dict) or news.get("status") == "error" or not (
        news.get("articles") or []
    ):
        return {
            "reply": NEWS_UNAVAILABLE.replace("this ticker", ticker),
            "out_of_scope": False,
            "mode": "research",
            "tools_used": tools_used,
            "repaired": False,
            "path": "news_empty",
        }

    news_raw = format_news_for_prompt(news)
    try:
        harness = harness_fn(
            ticker=ticker,
            news=news,
            news_raw=news_raw,
            llm=client,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            "live_news_harness_failed",
            level=logging.WARNING,
            step="research_live_news",
            status="error",
            data={"ticker": ticker, "error": str(exc)[:240]},
        )
        harness = {
            "news_summary": news_raw[:1200],
            "news_sentiment": "MIXED",
            "news_repaired": True,
        }

    tools_used.append("news_agent")
    sentiment = str(harness.get("news_sentiment") or "—")
    agent_note = str(harness.get("news_summary") or "")
    titles, urls = _article_allowlist(news)

    # Simple “latest headlines” asks: harness note + bullets is enough (~1 LLM).
    # Claim / judgment questions still get a Research synthesis pass.
    if _prefer_fast_headlines_only(text, history):
        reply = _deterministic_news_reply(
            ticker=ticker,
            sentiment=sentiment,
            note=agent_note,
            news=news,
            question=text,
        )
        out = {
            "reply": reply,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": tools_used,
            "repaired": False,
            "path": "live_news",
            "ticker": ticker,
            "fast_path": "harness_voice",
            "cache_hit": False,
        }
        _cache_put(cache_key, reply, out)
        log_event(
            logger,
            "live_news_done",
            step="research_live_news",
            status="ok",
            data={
                "ticker": ticker,
                "tools_used": tools_used,
                "sentiment": sentiment,
                "repaired": False,
                "articles": len(news.get("articles") or []),
                "fast_path": "harness_voice",
            },
        )
        return out

    repaired = False
    try:
        raw = _synthesize_with_timeout(
            question=text,
            ticker=ticker,
            sentiment=sentiment,
            agent_note=agent_note,
            news=news,
            history=history,
            llm=client,
        )
        if raw is None:
            raw = ""
            repaired = True
        cleaned = ensure_next_chips(
            polish_research_markdown(
                strip_brochure_voice(strip_tool_name_leaks(raw))
            )
        )
        ok, errors = _validate_live_news_reply(cleaned, titles=titles, urls=urls)
        if ok and _reply_fails_headline_grounding(cleaned, titles):
            ok = False
            errors = errors + ["headline_grounding"]
        if not ok:
            repaired = True
            reply = _deterministic_news_reply(
                ticker=ticker,
                sentiment=sentiment,
                note=agent_note,
                news=news,
                compact=_is_simple_headlines_ask(text),
                question=text,
            )
            log_event(
                logger,
                "live_news_repaired",
                step="research_live_news",
                status="ok",
                data={"ticker": ticker, "errors": errors},
            )
        else:
            reply = cleaned
            # Ensure live/source framing is present.
            if not re.search(rf"\*\*{re.escape(ticker)}\*\*", reply, re.I):
                reply = f"**{ticker}**\n\n{reply}"
                reply = ensure_next_chips(reply)
                repaired = True
    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            "live_news_synth_failed",
            level=logging.WARNING,
            step="research_live_news",
            status="error",
            data={"ticker": ticker, "error": str(exc)[:240]},
        )
        reply = _deterministic_news_reply(
            ticker=ticker,
            sentiment=sentiment,
            note=agent_note,
            news=news,
            question=text,
        )
        repaired = True

    reply = clean_assistant_markdown(reply)
    log_event(
        logger,
        "live_news_done",
        step="research_live_news",
        status="ok",
        data={
            "ticker": ticker,
            "tools_used": tools_used,
            "sentiment": sentiment,
            "repaired": repaired,
            "articles": len(news.get("articles") or []),
        },
    )
    out = {
        "reply": reply,
        "out_of_scope": False,
        "mode": "research",
        "tools_used": tools_used,
        "repaired": repaired,
        "path": "live_news",
        "ticker": ticker,
        "cache_hit": False,
    }
    _cache_put(cache_key, reply, out)
    return out
