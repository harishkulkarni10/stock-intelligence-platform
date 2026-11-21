"""Live news path for Research Co-Pilot — works with an empty desk.

Chat flow (Phase 1 — no Market harness):
1. Resolve ticker from the question (+ session context).
2. Fetch headlines via get_news (TTL cache in tools.py).
3. Simple headline asks → format only (**0 LLM**).
4. Claim check → **claim_zero_llm** when headlines do not support the claim; else one LLM (timeout + repair).
5. Topic follow-ups (e.g. Cybercab) → filter headlines; no full dump repeat.
6. Validate: no invented URLs/headlines; no buy/sell advice.

Analyze graph still uses run_news_harness in nodes.py — not this module.
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
from src.agents.assistant_voice import (
    clean_assistant_markdown,
    format_live_news_reply,
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
    r"bankrupt|bankruptcy|delist(?:ed|ing)?|delisting|"
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
    r"(?:tell me|give me|show me)\s+(?:the\s+)?(?:latest\s+)?(?:news|headlines)(?:\s+for\b|\s*$)|"
    r"(?:tell me|give me|show me)\s+headlines?\s+(?:for|on|about)\b|"
    r"what(?:'s| is)\s+happening\b|"
    r"[A-Za-z][A-Za-z0-9.&'\- ]{0,40}\s+news(?:\s+today)?\b"
    r").*$"
)
# Only these need an LLM to verify a yes/no claim against headlines.
_CLAIM_VERIFY_RE = re.compile(
    r"(?i)\b("
    r"wiped|bankrupt|bankruptcy|delist(?:ed|ing)?|delisting|halved|cut in half|"
    r"market cap|true or false|is it true|according to (?:the )?news"
    r")\b"
)
_PRIVATE_CO_EVENT_RE = re.compile(
    r"(?i)\b(ipo|went public|on the market|publicly traded|already trading|listed on)\b"
)
_HEADLINE_CHAT_RE = re.compile(
    r"(?i)\b("
    r"headlines?|latest\b|recent\b|coverage|"
    r"happening|what(?:'s| is)\s+(?:good|bad)|"
    r"one thing.*good|what.*bad|anything good|anything bad"
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
        "IPO",
        "SPAC",
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
    if _is_private_company_market_question(text):
        return True
    return False


_FOLLOW_UP_NEWS_RE = re.compile(
    r"(?i)\b("
    r"ceo|jensen|huang|what happened|what's with|why did|you said|"
    r"clarify|explain that|tell me more|that story|interim|go on"
    r")\b"
)
_TOPIC_FOLLOW_UP_RE = re.compile(
    r"(?i)\b(?:what about|how about|tell me more about)\b"
)
_TOPIC_TOKEN_STOP = frozenset(
    {
        "about",
        "that",
        "this",
        "those",
        "what",
        "with",
        "from",
        "your",
        "have",
        "does",
        "did",
        "the",
        "and",
        "for",
        "are",
        "was",
        "how",
        "story",
        "stories",
        "headline",
        "headlines",
        "news",
        "latest",
        "recent",
        "there",
        "they",
        "them",
        "when",
        "where",
        "which",
        "would",
        "could",
        "should",
        "just",
        "like",
        "mean",
        "said",
    }
)
_CLAIM_PLAIN_LABEL: dict[str, str] = {
    "bankruptcy": "bankruptcy or a Chapter 11 filing",
    "delisting": "delisting or removal from a major exchange",
    "market_cap_halved": "market cap being wiped out or cut roughly in half",
}
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
    if _TOPIC_FOLLOW_UP_RE.search(message):
        for turn in reversed(history[-LIVE_HISTORY_TURNS:]):
            blob = (turn.get("content") or "").lower()
            if any(marker in blob for marker in _LIVE_THREAD_MARKERS):
                return True
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


def _needs_claim_synthesis(question: str) -> bool:
    return bool(_CLAIM_VERIFY_RE.search(question or ""))


def _use_zero_llm_live_news(
    question: str,
    history: list[dict[str, str]] | None,
) -> bool:
    """Headline Q&A without Gemini — claims / verify questions are the exception."""
    if _needs_claim_synthesis(question):
        return False
    if _is_simple_headlines_ask(question):
        return True
    if _HEADLINE_CHAT_RE.search(question or ""):
        return True
    if is_live_news_follow_up(question, history):
        return True
    return False


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


def _private_company_focus_name(message: str) -> str | None:
    q_norm = _normalize_phrase(message or "")
    if "anthropic" in q_norm:
        return "anthropic"
    return None


def _is_private_company_market_question(message: str) -> bool:
    """IPO / listing asks about a private name — not a bogus IPO ticker."""
    if not _PRIVATE_CO_EVENT_RE.search(message or ""):
        return False
    return _private_company_focus_name(message) is not None


def _resolve_ticker_private_company_market(
    message: str,
    history: list[dict[str, str]] | None,
    *,
    active_ticker: str | None,
) -> str | None:
    """Pick a feed that may mention the private name — not merely the last symbol in chat."""
    focus = _private_company_focus_name(message)
    if not focus:
        return active_ticker.upper() if active_ticker else None

    hist = history or []
    for i in range(len(hist) - 1, -1, -1):
        turn = hist[i]
        if (turn.get("role") or "").strip().lower() != "assistant":
            continue
        if focus not in _normalize_phrase(turn.get("content") or ""):
            continue
        for j in range(i - 1, -1, -1):
            if (hist[j].get("role") or "").strip().lower() != "user":
                continue
            ticker = resolve_ticker_from_question(hist[j].get("content") or "")
            if ticker:
                return ticker

    if focus == "anthropic":
        for turn in reversed(hist):
            if (turn.get("role") or "").strip().lower() != "user":
                continue
            ticker = resolve_ticker_from_question(turn.get("content") or "")
            if ticker == "NVDA":
                return "NVDA"

    for turn in reversed(hist):
        if (turn.get("role") or "").strip().lower() != "user":
            continue
        ticker = resolve_ticker_from_question(turn.get("content") or "")
        if ticker:
            return ticker

    if active_ticker:
        return active_ticker.upper()
    return None


def _private_company_market_reply(
    *,
    ticker: str,
    question: str,
    news: dict[str, Any],
    sentiment: str,
) -> str:
    del sentiment
    focus = _private_company_focus_name(question) or "that company"
    titles = _headline_titles(news, max_items=10)
    matched = [t for t in titles if focus in t.lower()]
    q_norm = _normalize_phrase(question)
    ipo_ask = "ipo" in q_norm or bool(_PRIVATE_CO_EVENT_RE.search(question or ""))
    if not matched:
        opening = (
            f"**No** — nothing in this **{ticker}** headline pull mentions "
            f"**{focus.title()}** going public"
            + (" or an IPO." if ipo_ask else ".")
            + f" {focus.title()} remains private; stories below are **{ticker}** coverage."
        )
        show = titles
        max_items = 4
    else:
        if ipo_ask:
            opening = (
                f"**No IPO** — this pull mentions **{focus.title()}** in headline text "
                f"but not a listing or public debut. "
            ) + _topic_focus_opening(matched, [focus])
        else:
            opening = _topic_focus_opening(matched, [focus])
        show = matched
        max_items = min(6, max(3, len(matched)))
    body = format_live_news_reply(
        ticker=ticker,
        summary=opening,
        headline_titles=show,
        max_items=max_items,
    )
    return clean_assistant_markdown(f"{body}\n\n{_live_next_chips(ticker)}")


def resolve_ticker_from_context(
    message: str,
    history: list[dict[str, str]] | None = None,
    *,
    active_ticker: str | None = None,
) -> str | None:
    """Ticker from question, then recent chat, then active desk view."""
    if _is_private_company_market_question(message):
        return _resolve_ticker_private_company_market(
            message,
            history,
            active_ticker=active_ticker,
        )

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


_HEADLINE_BULLISH = re.compile(
    r"(?i)\b(beat|surge|soar|rally|record high|upgrade|wins?|wins|expands|growth)\b"
)
_HEADLINE_BEARISH = re.compile(
    r"(?i)\b(miss|plunge|slump|downgrade|layoff|bankrupt|delist|fired|probe|lawsuit)\b"
)


def _headline_sentiment(titles: list[str]) -> str:
    bull = bear = 0
    for title in titles:
        if _HEADLINE_BULLISH.search(title):
            bull += 1
        if _HEADLINE_BEARISH.search(title):
            bear += 1
    if bull > bear + 1:
        return "BULLISH"
    if bear > bull + 1:
        return "BEARISH"
    return "MIXED"


def _claims_in_question(question: str) -> list[str]:
    q = question or ""
    found: list[str] = []
    if re.search(r"(?i)\bbankrupt", q):
        found.append("bankruptcy")
    if re.search(r"(?i)\bdelist", q):
        found.append("delisting")
    if re.search(r"(?i)\b(market\s*cap|wiped|halved|cut in half)\b", q):
        found.append("market_cap_halved")
    return found


def _headlines_affirm_claim(claim: str, titles: list[str]) -> bool:
    blob = " ".join(titles).lower()
    if claim == "bankruptcy":
        return bool(re.search(r"\b(bankrupt|bankruptcy|chapter\s*11)\b", blob))
    if claim == "delisting":
        return bool(re.search(r"\b(delist(?:ed|ing)?|delisting)\b", blob))
    if claim == "market_cap_halved":
        return bool(
            re.search(
                r"(?i)(market\s*cap[^.]{0,40}(half|halv|wiped|50\s*%)|"
                r"wiped out|cut in half|halved)",
                blob,
            )
        )
    return False


def _any_headline_affirms_claims(claims: list[str], titles: list[str]) -> bool:
    return any(_headlines_affirm_claim(c, titles) for c in claims)


def _topic_keywords_for_reply(question: str | None) -> list[str]:
    """Narrow headlines only on explicit topic follow-ups (what about Cybercab…)."""
    q = (question or "").strip()
    if not q or not _TOPIC_FOLLOW_UP_RE.search(q):
        return []
    return _topic_keywords_from_question(q)


def _topic_keywords_from_question(question: str) -> list[str]:
    q = (question or "").strip()
    ql = q.lower()
    keywords: list[str] = []
    for m in re.finditer(r"«([^»]+)»|'([^']+)'|\"([^\"]+)\"", q):
        phrase = next(g for g in m.groups() if g)
        for w in re.findall(r"[a-z0-9]{3,}", phrase.lower()):
            if w not in _TOPIC_TOKEN_STOP:
                keywords.append(w)
    m = re.search(
        r"(?i)(?:what about|how about|tell me more about)\s+(?:the\s+)?(.+)",
        ql,
    )
    if m:
        tail = re.sub(r"\s*(stories?|news|headlines?)\s*\??\s*$", "", m.group(1))
        for w in re.findall(r"[a-z0-9]{3,}", tail):
            if w not in _TOPIC_TOKEN_STOP and w not in keywords:
                keywords.append(w)
    for w in re.findall(r"\b([A-Z][a-z]{2,}|[a-z]{4,})\b", q):
        wl = w.lower()
        if wl not in _TOPIC_TOKEN_STOP and wl not in keywords:
            keywords.append(wl)
    return keywords[:6]


def _filter_titles_by_keywords(
    titles: list[str],
    keywords: list[str],
) -> tuple[list[str], list[str]]:
    if not keywords:
        return titles, []
    matched: list[str] = []
    rest: list[str] = []
    hit_kw: set[str] = set()
    for title in titles:
        tl = title.lower()
        hits = [k for k in keywords if k in tl]
        if hits:
            matched.append(title)
            hit_kw.update(hits)
        else:
            rest.append(title)
    if matched:
        return matched + rest, sorted(hit_kw)
    return titles, []


def _topic_focus_opening(matched: list[str], keywords: list[str]) -> str:
    if not matched:
        return "No headlines in this pull match that topic by title."
    focus = keywords[0] if keywords else "that topic"
    lead = matched[0]
    if len(lead) > 88:
        lead = lead[:85] + "…"
    if len(matched) == 1:
        return (
            f"**1** headline in this pull matches **{focus}**: «{lead}»."
        )
    extra = len(matched) - 1
    return (
        f"**{len(matched)}** headlines mention **{focus}** "
        f"(lead: «{lead}»; {extra} more in the list)."
    )


def _deterministic_claim_reply(
    *,
    ticker: str,
    question: str,
    news: dict[str, Any],
    titles: list[str] | None = None,
) -> str:
    titles = titles or _headline_titles(news, max_items=10)
    claims = _claims_in_question(question)
    if claims:
        labels = [_CLAIM_PLAIN_LABEL.get(c, c) for c in claims]
        if len(labels) == 1:
            opening = (
                f"**No** — none of the headlines in this live pull support "
                f"{labels[0]}. Other themes dominate; closest stories below."
            )
        else:
            opening = (
                "**No** — this headline pull does not support the scenario in your "
                f"question ({'; '.join(labels)})."
            )
    else:
        opening = (
            "**No** — nothing in this live headline pull backs up that claim "
            "as stated. Closest stories below."
        )
    body = format_live_news_reply(
        ticker=ticker,
        summary=opening,
        headline_titles=titles,
        max_items=4,
    )
    return clean_assistant_markdown(f"{body}\n\n{_live_next_chips(ticker)}")


def _chat_opening_from_headlines(
    ticker: str,
    titles: list[str],
    sentiment: str,
) -> str:
    sym = (ticker or "").upper()
    if not titles:
        return f"No headline themes returned for {sym} in this fetch."
    lead = titles[0]
    if len(lead) > 96:
        lead = lead[:93] + "…"
    extra = max(0, len(titles) - 1)
    tail = f" Plus {extra} other stories in this pull." if extra else ""
    return (
        f"Headlines center on «{lead}».{tail} "
        f"Tone from wording feels {sentiment.lower()}."
    )


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


def _zero_llm_live_reply(
    *,
    ticker: str,
    sentiment: str,
    news: dict[str, Any],
    question: str | None = None,
    title_list: list[str] | None = None,
) -> str:
    titles = title_list if title_list is not None else _headline_titles(news, max_items=10)
    keywords = _topic_keywords_for_reply(question)
    ordered, hit_kw = _filter_titles_by_keywords(titles, keywords)
    matched = [t for t in ordered if any(k in t.lower() for k in hit_kw)] if hit_kw else []
    if hit_kw and matched:
        opening = _topic_focus_opening(matched, hit_kw)
        display = matched
        max_items = min(6, max(3, len(matched)))
    else:
        opening = _chat_opening_from_headlines(ticker, ordered, sentiment)
        display = ordered
        max_items = 6
    body = format_live_news_reply(
        ticker=ticker,
        summary=opening,
        headline_titles=display,
        max_items=max_items,
    )
    return clean_assistant_markdown(f"{body}\n\n{_live_next_chips(ticker)}")


def _deterministic_news_reply(
    *,
    ticker: str,
    sentiment: str,
    note: str,
    news: dict[str, Any],
    compact: bool = False,
    question: str | None = None,
    title_list: list[str] | None = None,
) -> str:
    del compact, note
    q = (question or "").strip()
    if _needs_claim_synthesis(q):
        return _deterministic_claim_reply(
            ticker=ticker,
            question=q,
            news=news,
            titles=title_list,
        )
    return _zero_llm_live_reply(
        ticker=ticker,
        sentiment=sentiment,
        news=news,
        question=question,
        title_list=title_list,
    )


def _is_simple_headlines_ask(question: str) -> bool:
    q = (question or "").strip()
    if not q or _needs_claim_synthesis(q):
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
    claim_block = ""
    if _needs_claim_synthesis(question):
        claim_block = """
CLAIM CHECK (required):
- First line after **{ticker}**: start with **Yes**, **No**, or **Unclear** — does this live pull support the user's claim?
- If **No**, say clearly nothing in the listed headlines supports it; do not substitute a generic theme essay.
- Cite only headline facts; no invented events.
""".format(
            ticker=ticker
        )
    system = f"""Desk assistant — concise chat answer about {ticker} from LIVE NEWS only.
{claim_block}
STYLE:
- **{ticker}** then 2–3 short sentences (themes only) unless CLAIM CHECK applies (then Yes/No first).
- **Worth a look** — 4–6 bullets; label: one-line fact (e.g. "Berlin expansion: …").
- No provider names, timestamps, disclaimers, or stock phrases like "navigating a period of intense".
- Use CONVERSATION only when the user is following up.

RULES: Only facts in the expert note and headline list. No new numbers. No buy/sell.
Never mention CEO departures, receivables, or deals unless a headline line states them.
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
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
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
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


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
    if ("resign" in low or "departure" in low or "interim ceo" in low) and (
        "resign" not in blob and "depart" not in blob and "interim" not in blob
    ):
        return True
    if "receivable" in low and "receivable" not in blob:
        return True
    if "anthropic" in low and "anthropic" not in blob:
        return True
    for m in re.finditer(r"\$\s*(\d+(?:\.\d+)?)\s*b", low):
        if m.group(1) not in blob and "billion" not in blob:
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
    client = llm if llm is not None else get_chat_llm()
    llm_calls = 0

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
    title_list = _headline_titles(news, max_items=10)
    sentiment = _headline_sentiment(title_list)
    context_note = _chat_opening_from_headlines(ticker, title_list, sentiment)
    titles, urls = _article_allowlist(news)
    news_fetch_cache_hit = bool(news.get("cache_hit"))

    def _out_base(**extra: Any) -> dict[str, Any]:
        base = {
            "out_of_scope": False,
            "mode": "research",
            "path": "live_news",
            "ticker": ticker,
            "cache_hit": False,
            "news_fetch_cache_hit": news_fetch_cache_hit,
            "llm_calls": llm_calls,
        }
        base.update(extra)
        return base

    if _is_private_company_market_question(text):
        reply = _private_company_market_reply(
            ticker=ticker,
            question=text,
            news=news,
            sentiment=sentiment,
        )
        out = _out_base(
            reply=reply,
            tools_used=tools_used,
            repaired=False,
            fast_path="private_co_zero_llm",
            llm_calls=0,
        )
        _cache_put(cache_key, reply, out)
        log_event(
            logger,
            "live_news_done",
            step="research_live_news",
            status="ok",
            data={
                "ticker": ticker,
                "tools_used": tools_used,
                "fast_path": "private_co_zero_llm",
                "news_fetch_cache_hit": news_fetch_cache_hit,
                "llm_calls": 0,
            },
        )
        return out

    # Fact-check claims when headlines do not contain supporting terms (0 LLM).
    if _needs_claim_synthesis(text):
        claims = _claims_in_question(text)
        if claims and not _any_headline_affirms_claims(claims, title_list):
            reply = _deterministic_claim_reply(
                ticker=ticker,
                question=text,
                news=news,
                titles=title_list,
            )
            out = _out_base(
                reply=reply,
                tools_used=tools_used,
                repaired=False,
                fast_path="claim_zero_llm",
                llm_calls=0,
            )
            _cache_put(cache_key, reply, out)
            log_event(
                logger,
                "live_news_done",
                step="research_live_news",
                status="ok",
                data={
                    "ticker": ticker,
                    "tools_used": tools_used,
                    "claims": claims,
                    "fast_path": "claim_zero_llm",
                    "news_fetch_cache_hit": news_fetch_cache_hit,
                    "llm_calls": 0,
                },
            )
            return out

    # Simple headline asks — deterministic assistant voice, no LLM.
    if _use_zero_llm_live_news(text, history):
        reply = _zero_llm_live_reply(
            ticker=ticker,
            sentiment=sentiment,
            news=news,
            question=text,
            title_list=title_list,
        )
        out = _out_base(
            reply=reply,
            tools_used=tools_used,
            repaired=False,
            fast_path="zero_llm",
            llm_calls=0,
        )
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
                "fast_path": "zero_llm",
                "news_fetch_cache_hit": news_fetch_cache_hit,
                "llm_calls": 0,
            },
        )
        return out

    repaired = False
    llm_calls = 1
    try:
        raw = _synthesize_with_timeout(
            question=text,
            ticker=ticker,
            sentiment=sentiment,
            agent_note=news_raw[:1200],
            news=news,
            history=history,
            llm=client,
        )
        if raw is None:
            reply = _deterministic_news_reply(
                ticker=ticker,
                sentiment=sentiment,
                note=context_note,
                news=news,
                question=text,
                title_list=title_list,
            )
            fb_fast = (
                "claim_zero_llm"
                if _needs_claim_synthesis(text)
                else "zero_llm"
            )
            out = _out_base(
                reply=reply,
                tools_used=tools_used,
                repaired=True,
                fast_path=fb_fast,
                llm_calls=0,
            )
            _cache_put(cache_key, reply, out)
            return out
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
                note=context_note,
                news=news,
                question=text,
                title_list=title_list,
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
            if _needs_claim_synthesis(text) and not re.search(
                r"(?i)\*\*(yes|no|unclear)\*\*", reply
            ):
                repaired = True
                reply = _deterministic_claim_reply(
                    ticker=ticker,
                    question=text,
                    news=news,
                    titles=title_list,
                )
            elif not re.search(rf"\*\*{re.escape(ticker)}\*\*", reply, re.I):
                # Ensure live/source framing is present.
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
            note=context_note,
            news=news,
            question=text,
            title_list=title_list,
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
            "llm_calls": llm_calls,
            "news_fetch_cache_hit": news_fetch_cache_hit,
        },
    )
    out = _out_base(
        reply=reply,
        tools_used=tools_used,
        repaired=repaired,
        fast_path="synth_one_llm" if llm_calls else "repair_zero_llm",
        llm_calls=llm_calls,
    )
    _cache_put(cache_key, reply, out)
    return out
