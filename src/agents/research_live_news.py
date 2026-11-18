"""Live news path for Research Co-Pilot — works with an empty desk.

Flow:
1. Resolve ticker from the user question (symbol or company name).
2. Fetch headlines via get_news (same tool the Market expert uses).
3. Run the news harness (Market expert) for sentiment + specialist note.
4. Have Research LLM read that agent output + headlines and answer the user.
5. Validate: no invented URLs/headlines; no buy/sell advice.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from logger.logger import get_logger, log_event
from src.agents.llm import get_chat_llm, message_text
from src.agents.news_guardrails import run_news_harness
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
from src.agents.tools import format_news_for_prompt, get_news
from src.data.ingestion import normalize_ticker

logger = get_logger("sip.research_live_news")

_NEWS_HINT_RE = re.compile(
    r"(?i)\b("
    r"news|headline|headlines|article|articles|press|coverage|"
    r"sentiment|what(?:'s| is) (?:going on|happening)|"
    r"latest (?:on|about)|recent (?:news|headlines|coverage)"
    r")\b"
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
    return bool(_NEWS_HINT_RE.search(message or ""))


def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


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


def _deterministic_news_reply(
    *,
    ticker: str,
    sentiment: str,
    note: str,
    news: dict[str, Any],
) -> str:
    titles, _ = _article_allowlist(news)
    bullets = "\n".join(f"- {t}" for t in titles[:6]) or "- No headlines kept."
    note_clip = (note or "").strip()
    if len(note_clip) > 900:
        note_clip = note_clip[:899] + "…"
    return (
        f"### Live news · {ticker}\n"
        f"Source: live fetch (not a desk run packet). "
        f"Headline tone from the Market expert: {sentiment or '—'}.\n\n"
        f"### Specialist note\n"
        f"{note_clip or 'No specialist note produced.'}\n\n"
        f"### Headlines used\n"
        f"{bullets}\n\n"
        f"### Caveat\n"
        f"Research support only — not trade advice. "
        f"This is live news, not a full multi-agent analyze run.\n\n"
        f"{_DEFAULT_NEXT}"
    )


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
    titles, urls = _article_allowlist(news)
    articles_json = json.dumps(
        [
            {
                "headline": a.get("headline") or a.get("title"),
                "url": a.get("url") or a.get("link"),
                "published": a.get("published") or a.get("datetime") or a.get("date"),
                "scope": a.get("scope"),
            }
            for a in (news.get("articles") or [])[:10]
            if isinstance(a, dict)
        ],
        ensure_ascii=False,
        indent=2,
    )
    system = f"""You are Research Co-Pilot for Stock Intelligence.

The user asked a NEWS question. There may be no desk analyze run.
You are given LIVE tool results: Market expert note + headlines for {ticker}.

Rules:
- Answer ONLY from LIVE NEWS FACTS below. Do not invent headlines, URLs, or figures.
- Prefer the Market expert note for tone/sentiment; use headlines as support.
- Say clearly this is live news, not a finished desk packet.
- ### headings and - bullets. No bold/italics.
- No buy/sell advice.
- End with:
{_DEFAULT_NEXT}
"""
    messages: list = [SystemMessage(content=system)]
    for turn in (history or [])[-4:]:
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            messages.append(HumanMessage(content=f"ASSISTANT_PREV:\n{content}"))
        else:
            messages.append(HumanMessage(content=content))
    user = (
        f"QUESTION:\n{question}\n\n"
        f"LIVE NEWS FACTS\n"
        f"Ticker: {ticker}\n"
        f"Market expert sentiment: {sentiment}\n"
        f"Market expert note:\n{agent_note}\n\n"
        f"Headlines JSON:\n{articles_json}\n\n"
        f"Allowed titles: {titles[:12]}\n"
        f"Allowed URLs: {urls[:12]}\n\n"
        "Write the final ### markdown answer now."
    )
    messages.append(HumanMessage(content=user))
    return message_text(llm.invoke(messages)).strip()


def answer_live_news_question(
    message: str,
    history: list[dict[str, str]] | None = None,
    *,
    llm=None,
    news_fetcher=None,
    news_harness=None,
) -> dict[str, Any]:
    """Live news research path for empty-desk (or off-desk) questions."""
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

    ticker = resolve_ticker_from_question(text)
    if not ticker:
        return {
            "reply": NEED_TICKER_NEWS,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
            "path": "need_ticker",
        }

    fetch = news_fetcher or get_news
    harness_fn = news_harness or run_news_harness
    client = llm if llm is not None else get_chat_llm()

    try:
        news = fetch(ticker, limit=6)
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
    if _is_simple_headlines_ask(text):
        reply = _deterministic_news_reply(
            ticker=ticker,
            sentiment=sentiment,
            note=agent_note,
            news=news,
        )
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
                "fast_path": "harness_only",
            },
        )
        return {
            "reply": reply,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": tools_used,
            "repaired": False,
            "path": "live_news",
            "ticker": ticker,
            "fast_path": "harness_only",
        }

    repaired = False
    try:
        raw = _synthesize_from_agent(
            question=text,
            ticker=ticker,
            sentiment=sentiment,
            agent_note=agent_note,
            news=news,
            history=history,
            llm=client,
        )
        cleaned = ensure_next_chips(
            polish_research_markdown(
                strip_brochure_voice(strip_tool_name_leaks(raw))
            )
        )
        ok, errors = _validate_live_news_reply(cleaned, titles=titles, urls=urls)
        if not ok:
            repaired = True
            reply = _deterministic_news_reply(
                ticker=ticker,
                sentiment=sentiment,
                note=agent_note,
                news=news,
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
            if "live" not in reply.lower():
                reply = (
                    f"### Live news · {ticker}\n"
                    f"Source: live fetch (not a desk run packet).\n\n"
                    + reply
                )
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
        )
        repaired = True

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
    return {
        "reply": reply,
        "out_of_scope": False,
        "mode": "research",
        "tools_used": tools_used,
        "repaired": repaired,
        "path": "live_news",
        "ticker": ticker,
    }
