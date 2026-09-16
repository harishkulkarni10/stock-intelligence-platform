"""Grounded product guide chat — Google-style structured answers."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.llm import get_chat_llm, message_text

KNOWLEDGE_PATH = Path(__file__).with_name("help_knowledge.md")

OUT_OF_SCOPE = (
    "Out of scope — I only cover how this research desk and its agents work, "
    "not ticker picks, prices, or trading advice."
)

SAFE_OVERVIEW = """### What this desk is
Stock Intelligence is an equity research desk. Search a ticker, run analysis, and get one packet: forecast chart, company overview, three specialist notes, and Sources.

### What a run includes
- Forecast chart — history plus a short projected path
- Performance analyst — what the path implies (trend, confidence, caveats)
- Market expert — news tone grounded in headlines under Sources
- Financial analyst — fundamentals health, strengths, weaknesses
- On your desk — reopen finished tickers from this visit only

### How to use it
- Compare trend, news, and financials together — disagreement is useful
- Check important claims against the chart and Sources
- Research support only — no buy or sell advice

NEXT: How to read a run|Performance analyst|Market expert|Financial analyst"""

MAX_HISTORY = 6

_BANNED = re.compile(
    r"\b("
    r"redis|lstm|yfinance|finnhub|ollama|langgraph|langchain|gemma|openai|"
    r"force_refresh|guardrail_ok|analyze-perf|analyze_cache|ttl|"
    r"env\s*var|api\s*endpoint|api\s*field|request\s*field|"
    r"prompt[- ]validate|non-?llm|code-based|yahoo/?yfinance|yahoo\s+finance|"
    r"child\s+lstm|parent\s+model|persistence\s+baseline|guardrails?"
    r")\b",
    re.IGNORECASE,
)


def load_knowledge() -> str:
    try:
        return KNOWLEDGE_PATH.read_text(encoding="utf-8")
    except OSError:
        return SAFE_OVERVIEW


def _normalize_structure(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").strip()
    cleaned = re.sub(r"`+", "", cleaned)
    # Strip markdown bold/italic markers the UI does not render as markdown.
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_system_prompt(knowledge: str | None = None) -> str:
    pack = knowledge if knowledge is not None else load_knowledge()
    return f"""You are Guide — the in-app AI for Stock Intelligence.

Goal: help users understand this product deeply enough to use it well.
Voice: clear Google-product support tone. Structured. Useful. No fluff intros.

OUTPUT FORMAT (unless Out of scope):
### Title
One clear sentence.

### Section
- Concrete bullet
- Concrete bullet

Use 3–5 sections when explaining an agent or the full desk.
Use enough bullets to teach the feature (typically 6–12 bullets across the answer).
Include ### Tip only when it adds a practical caution.
End with optional follow-ups:
NEXT: Label one|Label two|Label three

Formatting rules (strict):
- Use ### headings and - bullets only.
- Never use markdown bold or italics. Never write ** or __ or *emphasis*.
- Never wrap labels like **Agent roles:** — write ### Agent roles instead.
- Do not pad with empty encouragement.

Depth rules:
- Prefer teaching over slogans. Say what the agent reads, writes, which labels appear, and how the user should check the claim.
- Stay scannable: short bullets, not long paragraphs (1–2 sentences per bullet max).
- Answer only from PRODUCT KNOWLEDGE.
- Never name databases, servers, models, libraries, vendors, env vars, or API fields.
- Never say Redis, LSTM, Yahoo, yfinance, guardrails, or similar.

If unrelated (companies, ticker picks, prices, trading advice), reply exactly:
{OUT_OF_SCOPE}

Good example for "What can I ask you?":
### What you can ask
Questions about how this research desk works.

### Agent roles
- What Performance, Market expert, or Financial analyst do
- What their labels mean and how to check them

### Using the desk
- Forecast chart, On your desk, Refresh analysis
- How to read a finished run when the tabs disagree

### Out of scope
- No ticker picks, live prices, or trading advice
- No company trivia unrelated to this desk

NEXT: Performance analyst|How to read a run|On your desk

PRODUCT KNOWLEDGE:
{pack}
"""

def _looks_like_overview(question: str) -> bool:
    q = question.lower()
    return any(
        phrase in q
        for phrase in (
            "about the app",
            "about this app",
            "about the application",
            "about this application",
            "what is this",
            "what does this",
            "tell me about",
            "how does this work",
            "what is stock intelligence",
        )
    )


def sanitize_help_reply(reply: str, *, question: str = "") -> tuple[str, bool]:
    """Return (clean_reply, out_of_scope). Keeps light markdown for the UI."""
    text = _normalize_structure(reply)
    if not text:
        return OUT_OF_SCOPE, True
    if text.lower().startswith("out of scope"):
        return OUT_OF_SCOPE, True
    if not _BANNED.search(text):
        return text, False

    if _looks_like_overview(question):
        return SAFE_OVERVIEW, False

    q = question.lower()
    if any(word in q for word in ("cache", "saved", "refresh", "recent result")):
        return (
            "### Recent saved results\n"
            "If you run the same ticker again soon, you may see a prior result faster.\n\n"
            "### What you'll notice\n"
            "- A badge on results when a recent saved result is shown\n"
            "- Refresh analysis starts a brand-new run now\n"
            "- On your desk is separate — it only keeps finished tickers from this visit\n\n"
            "### Tip\n"
            "On your desk clears when you refresh or leave the page.\n\n"
            "NEXT: On your desk|How to read a run|What this desk is",
            False,
        )

    parts = re.split(r"(?<=[.!?])\s+", text)
    kept = [p for p in parts if p and not _BANNED.search(p)]
    cleaned = " ".join(kept).strip()
    if len(cleaned) >= 40:
        return cleaned, False
    return SAFE_OVERVIEW, False


def answer_help_question(
    message: str,
    history: list[dict[str, str]] | None = None,
    *,
    llm=None,
) -> dict[str, str | bool]:
    """Return `{ reply, out_of_scope }` for a product-guide turn."""
    text = (message or "").strip()
    if not text:
        return {
            "reply": SAFE_OVERVIEW,
            "out_of_scope": False,
        }

    system = build_system_prompt()
    messages: list = [SystemMessage(content=system)]
    for turn in (history or [])[-MAX_HISTORY:]:
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    messages.append(HumanMessage(content=text))

    client = llm if llm is not None else get_chat_llm()
    try:
        raw = message_text(client.invoke(messages)).strip()
    except Exception:  # noqa: BLE001
        return {
            "reply": "Guide is temporarily unavailable. Try again in a moment.",
            "out_of_scope": False,
        }

    reply, out = sanitize_help_reply(raw, question=text)
    return {"reply": reply, "out_of_scope": out}
