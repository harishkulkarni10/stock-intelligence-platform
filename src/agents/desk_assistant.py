"""Unified desk assistant — one bot for product help, desk packets, and live news."""

from __future__ import annotations

import re
from typing import Any

from src.agents.help_chat import OUT_OF_SCOPE as GUIDE_OOS, answer_help_question
from src.agents.research_chat import answer_research_question

# Product / UI how-to — not ticker packet Q&A.
_PRODUCT_GUIDE_RE = re.compile(
    r"(?i)\b("
    r"what does the (?:performance|market|financial|risk) analyst do|"
    r"what does the (?:market expert|research brief) do|"
    r"how do i read (?:a )?(?:finished )?run|"
    r"how to read (?:a )?(?:finished )?run|"
    r"what is on your desk|"
    r"what(?:'s| is) (?:on your desk|the lab)|"
    r"how does (?:this|the) (?:desk|app|application) work|"
    r"what is stock intelligence|"
    r"what (?:is|does) this desk|"
    r"tell me about (?:the )?(?:app|application|desk)\b|"
    r"what (?:can|should) i ask"
    r")\b"
)

# Packet/desk Q&A — never send to Guide-only path.
_PACKET_INTENT_RE = re.compile(
    r"(?i)\b("
    r"explain disagreement|walk through risk|why this confidence|"
    r"forecast path|projected move|compare .{0,40} (?:move|risk)|"
    r"bull vs bear|summarize the research|"
    r"news|headline|bankrupt|delist|market cap|wiped|"
    r"latest|recent|coverage|sentiment|confidence on|"
    r"disagree|stance on|financial health of|revenue|margin"
    r")\b"
)


def is_product_guide_question(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return True
    if _PACKET_INTENT_RE.search(text):
        return False
    return bool(_PRODUCT_GUIDE_RE.search(text))


def answer_desk_assistant(
    message: str,
    history: list[dict[str, str]] | None = None,
    run_context: dict[str, Any] | None = None,
    *,
    llm=None,
) -> dict[str, Any]:
    """Single entry for /help-chat — routes guide vs research/live internally."""
    text = (message or "").strip()
    if is_product_guide_question(text):
        guide = answer_help_question(text, history, llm=llm)
        return {
            "reply": guide.get("reply") or "",
            "out_of_scope": bool(guide.get("out_of_scope")),
            "mode": "assistant",
            "tools_used": [],
            "repaired": False,
            "path": "product_guide",
        }

    result = answer_research_question(
        text,
        history,
        run_context,
        llm=llm,
    )
    result["mode"] = "assistant"
    return result