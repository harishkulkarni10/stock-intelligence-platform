"""Shared formatting for the unified desk assistant — clear, short replies."""

from __future__ import annotations

import re

# Strip noisy sections users do not need in chat.
_NOISE_SECTIONS = re.compile(
    r"(?is)"
    r"###\s*Tip\s*\n.*?(?=\n###\s|\nNEXT:|\Z)"
    r"|###\s*Desk packets\s*\n.*?(?=\n###\s|\nNEXT:|\Z)"
    r"|###\s*Caveat\s*\n.*?(?=\n###\s|\nNEXT:|\Z)"
    r"|###\s*Summary\s*\n(?:Source:|\s*Headline tone:).*?(?=\n###\s|\n\*\*|\nNEXT:|\Z)"
)
_DISCLAIMER_LINES = re.compile(
    r"(?im)^(?:not investment advice|research support only|not trade advice)"
    r".*$\n?"
)
_FEED_META = re.compile(
    r"(?im)^(?:updated\s+\d{4}-\d{2}-\d{2}|source:\s*live fetch|"
    r"yahoo feed|finnhub).*$\n?"
)
_ISO_UPDATED = re.compile(
    r"(?im)^\s*updated\s+\d{4}-\d{2}-\d{2}T[\d:Z.]+\..*$\n?"
)
_SENTIMENT_ONLY = re.compile(r"(?im)^\s*\*\*?sentiment:\*\*?\s*\S+\s*$")


def clip_prose(text: str, *, max_sentences: int = 3, max_chars: int = 520) -> str:
    """First few sentences for chat openings."""
    t = (text or "").strip()
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = " ".join(parts[:max_sentences]).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return out


def harness_analysis_for_chat(note: str) -> str:
    """Pull the Analysis body out of the Market-expert harness text."""
    n = strip_market_harness_preface(note)
    if re.search(r"(?i)\banalysis:\s*", n):
        n = re.split(r"(?i)analysis:\s*", n, maxsplit=1)[-1]
    n = re.split(r"(?i)implications:\s*", n, maxsplit=1)[0]
    n = re.sub(r"(?im)^sentiment:\s*\S+\s*$", "", n)
    return clip_prose(n.strip(), max_sentences=4, max_chars=560)


def harness_to_chat_reply(
    *,
    ticker: str,
    harness_note: str,
    headline_titles: list[str],
    sentiment: str | None = None,
) -> str:
    summary = harness_analysis_for_chat(harness_note)
    if sentiment and summary and "mixed" not in summary.lower():
        summary = f"{summary.rstrip()} Tone from headlines feels {sentiment.lower()}."
    return format_live_news_reply(
        ticker=ticker,
        summary=summary,
        headline_titles=headline_titles,
        max_items=6,
    )


def strip_market_harness_preface(note: str) -> str:
    """Turn news-agent blob into prose without harness labels."""
    n = (note or "").replace("\r\n", "\n").strip()
    n = re.sub(r"(?is)^sentiment:\s*\S+\s*\n*", "", n)
    n = re.sub(r"(?i)^analysis:\s*", "", n)
    n = re.sub(r"\n{3,}", "\n\n", n)
    return n.strip()


def format_live_news_reply(
    *,
    ticker: str,
    summary: str,
    headline_titles: list[str],
    max_items: int = 6,
) -> str:
    """Readable live-news answer — summary first, short list of stories."""
    sym = (ticker or "").strip().upper()
    body = (strip_market_harness_preface(summary) or "").strip()
    if not body:
        body = "Headlines are in; nothing strong enough to summarize yet."
    lines = [ln.strip() for ln in headline_titles if (ln or "").strip()]
    lines = lines[:max_items]
    parts = [f"**{sym}** — {body}"]
    if lines:
        bullets = "\n".join(f"- {title}" for title in lines)
        parts.append(f"**Worth a look**\n{bullets}")
    return "\n\n".join(parts)


def clean_assistant_markdown(text: str) -> str:
    """Remove boilerplate blocks; keep structure readable in the chat UI."""
    cleaned = (text or "").replace("\r\n", "\n").strip()
    cleaned = _NOISE_SECTIONS.sub("\n\n", cleaned)
    cleaned = _DISCLAIMER_LINES.sub("", cleaned)
    cleaned = _FEED_META.sub("", cleaned)
    cleaned = _ISO_UPDATED.sub("", cleaned)
    cleaned = _SENTIMENT_ONLY.sub("", cleaned)
    cleaned = re.sub(
        r"(?im)^\s*###\s+([A-Z]{1,5})\s+—\s*live news\s*$",
        r"**\1**",
        cleaned,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" — ", " — ", cleaned)  # normalize em dash spacing
    return cleaned.strip()


def short_missing_ticker_reply(*, missing: str, active: str | None = None) -> str:
    sym = (missing or "that symbol").upper()
    act = (active or "").strip().upper()
    lead = f"No finished analyze packet for **{sym}** on this visit."
    if act and act != sym:
        lead += f" The open desk view is **{act}**."
    return (
        f"{lead}\n\n"
        f"You can still ask **live headlines for {sym}**, or run analyze and ask again.\n\n"
        f"NEXT: Latest headlines for {sym}|Explain disagreement"
    )