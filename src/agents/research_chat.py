"""Research Co-Pilot — desk-scoped Q&A over finished analyze packets."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from logger.logger import get_logger, log_event
from src.agents.llm import get_chat_llm, message_text
from src.agents.research_guardrails import (
    ADVICE_META,
    NO_RUN,
    OUT_OF_SCOPE,
    ResearchFacts,
    build_research_facts,
    build_research_system_prompt,
    deterministic_research_answer,
    ensure_next_chips,
    is_advice_meta_question,
    is_advice_question,
    is_compare_moves_question,
    is_multi_risk_question,
    is_open_run_meta_question,
    missing_desk_ticker_reply,
    multi_packet_compare_moves_reply,
    multi_packet_risk_reply,
    open_run_meta_reply,
    polish_research_markdown,
    prefers_deterministic_answer,
    repair_research_reply,
    strip_brochure_voice,
    strip_tool_name_leaks,
    strip_ungrounded_catalyst_sentences,
    validate_research_reply,
)
from src.agents.research_tools import (
    DeskContext,
    execute_tool_plan,
    is_multi_ticker_question,
    mentioned_desk_tickers,
    normalize_desk_context,
    off_desk_tickers_requested,
    resolve_target_tickers,
    run_tool,
)

logger = get_logger("sip.research_chat")

MAX_HISTORY = 6
MAX_TOOL_ROUNDS = 4
_TOOL_CALL_RE = re.compile(
    r"\{[^{}]*\"tool\"\s*:\s*\"([a-z_]+)\"[^{}]*\}",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_structure(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").strip()
    cleaned = re.sub(r"`+", "", cleaned)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    raw = (text or "").strip()
    if not raw.startswith("{") and "\"tool\"" not in raw:
        return None
    # Prefer first JSON object in the reply.
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        match = _TOOL_CALL_RE.search(raw)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    else:
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            match = _TOOL_CALL_RE.search(raw)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    name = str(payload.get("tool") or "").strip()
    if not name:
        return None
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    return name, args


def _history_messages(history: list[dict[str, str]] | None) -> list:
    messages: list = []
    for turn in (history or [])[-MAX_HISTORY:]:
        role = (turn.get("role") or "").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def _tool_block_for_targets(
    desk: DeskContext,
    targets: list[str],
    question: str,
) -> tuple[list[str], str, dict[str, Any]]:
    """Run tools on primary target; add overviews for any extra named tickers."""
    primary_sym = targets[0]
    primary = desk.get(primary_sym)
    assert primary is not None
    tools_used, tool_block = execute_tool_plan(primary, question=question)
    if len(targets) > 1:
        extra_blocks: list[str] = [tool_block]
        for sym in targets[1:]:
            pkt = desk.get(sym)
            if not pkt:
                continue
            extra_tools, overview_block = execute_tool_plan(
                pkt, [("get_run_overview", {}), ("compare_labels", {})]
            )
            tools_used.extend(f"{sym}:{name}" for name in extra_tools)
            extra_blocks.append(f"### DESK PACKET {sym}\n{overview_block}")
        tool_block = "\n\n".join(extra_blocks)
    return tools_used, tool_block, primary


def _synthesize(
    *,
    question: str,
    tool_block: str,
    facts: ResearchFacts,
    history: list[dict[str, str]] | None,
    llm,
    desk: DeskContext,
) -> str:
    system = build_research_system_prompt(
        facts,
        desk_tickers=desk.tickers,
        active_ticker=desk.active_ticker,
    )
    messages: list = [SystemMessage(content=system)]
    messages.extend(_history_messages(history))
    user = (
        f"QUESTION:\n{question}\n\n"
        f"DESK TICKERS: {', '.join(desk.tickers)}\n"
        f"ACTIVE VIEW: {desk.active_ticker}\n"
        f"PRIMARY TARGET: {facts.ticker}\n\n"
        f"PACKET FACTS:\n{tool_block}\n\n"
        "Write the final ### markdown answer from PACKET FACTS only. "
        "Do not name tools. Do not invent figures. "
        "If you truly need another packet slice, reply with JSON only: "
        '{"tool": "compare_labels", "args": {}}'
    )
    messages.append(HumanMessage(content=user))
    return message_text(llm.invoke(messages)).strip()


def answer_research_question(
    message: str,
    history: list[dict[str, str]] | None = None,
    run_context: dict[str, Any] | None = None,
    *,
    llm=None,
) -> dict[str, Any]:
    """Return `{ reply, out_of_scope, mode, tools_used, repaired }`."""
    text = (message or "").strip()
    desk = normalize_desk_context(run_context)

    if not desk:
        return {
            "reply": NO_RUN,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
        }

    named = mentioned_desk_tickers(text, desk) if text else []
    targets = named if named else (
        resolve_target_tickers(text, desk) if text else [desk.active_ticker]
    )
    packet = desk.get(targets[0]) if targets else desk.get(desk.active_ticker)
    if not packet:
        return {
            "reply": NO_RUN,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
        }

    if not text:
        overview_tools, _ = execute_tool_plan(
            packet, [("get_run_overview", {}), ("compare_labels", {})]
        )
        reply = deterministic_research_answer(packet)
        return {
            "reply": reply,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": overview_tools,
            "repaired": False,
        }

    if is_advice_meta_question(text):
        return {
            "reply": ADVICE_META,
            "out_of_scope": True,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
        }

    if is_advice_question(text):
        log_event(
            logger,
            "research_out_of_scope",
            step="research_chat",
            status="ok",
            data={"reason": "advice_question", "ticker": packet.get("ticker")},
        )
        return {
            "reply": OUT_OF_SCOPE,
            "out_of_scope": True,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
        }

    off_desk = off_desk_tickers_requested(text, desk)

    # Multi-ticker compare / risk — scripted from every named desk packet.
    if is_multi_ticker_question(text, named) and (
        is_compare_moves_question(text) or is_multi_risk_question(text) or len(named) >= 2
    ):
        if is_compare_moves_question(text) or (
            len(named) >= 2
            and any(w in text.lower() for w in ("projected", "move", "compare", "mean"))
            and "risk" not in text.lower()
        ):
            reply = multi_packet_compare_moves_reply(
                desk, named, missing=off_desk
            )
            path = "multi_compare_moves"
        elif is_multi_risk_question(text) or (
            len(named) >= 2 and "risk" in text.lower()
        ):
            reply = multi_packet_risk_reply(desk, named, missing=off_desk)
            path = "multi_risk"
        else:
            reply = None
            path = ""
        if reply:
            tools_used = [f"get_run_overview:{t}" for t in named]
            log_event(
                logger,
                "research_chat_done",
                step="research_chat",
                status="ok",
                data={
                    "ticker": desk.active_ticker,
                    "tools_used": tools_used,
                    "repaired": False,
                    "path": path,
                    "targets": named,
                    "missing": off_desk,
                },
            )
            return {
                "reply": reply,
                "out_of_scope": False,
                "mode": "research",
                "tools_used": tools_used,
                "repaired": False,
            }

    # Off-desk only when nothing on-desk was resolved (never miss an available ticker).
    if off_desk and not named:
        reply = missing_desk_ticker_reply(
            active_ticker=desk.active_ticker,
            desk_tickers=desk.tickers,
            mentioned=off_desk,
        )
        log_event(
            logger,
            "research_chat_done",
            step="research_chat",
            status="ok",
            data={
                "ticker": desk.active_ticker,
                "tools_used": [],
                "repaired": False,
                "path": "missing_desk_ticker",
                "mentioned": off_desk,
            },
        )
        return {
            "reply": reply,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": [],
            "repaired": False,
        }

    if is_open_run_meta_question(text):
        reply = open_run_meta_reply(
            packet,
            desk_tickers=desk.tickers,
            active_ticker=desk.active_ticker,
        )
        log_event(
            logger,
            "research_chat_done",
            step="research_chat",
            status="ok",
            data={
                "ticker": packet.get("ticker"),
                "tools_used": ["get_run_overview"],
                "repaired": False,
                "path": "open_run_meta",
                "desk_tickers": desk.tickers,
            },
        )
        return {
            "reply": reply,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": ["get_run_overview"],
            "repaired": False,
        }

    # Glossary / confidence / stance-why: scripted copy beats soft LLM myths.
    if prefers_deterministic_answer(text):
        tools_used, _, packet = _tool_block_for_targets(desk, targets, text)
        reply = deterministic_research_answer(packet, question=text)
        log_event(
            logger,
            "research_chat_done",
            step="research_chat",
            status="ok",
            data={
                "ticker": packet.get("ticker"),
                "tools_used": tools_used,
                "repaired": False,
                "path": "deterministic_glossary",
                "targets": targets,
            },
        )
        return {
            "reply": reply,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": tools_used,
            "repaired": False,
        }

    tools_used, tool_block, packet = _tool_block_for_targets(desk, targets, text)
    facts = build_research_facts(packet)
    client = llm if llm is not None else get_chat_llm()

    raw = ""
    try:
        for _round in range(MAX_TOOL_ROUNDS):
            raw = _synthesize(
                question=text,
                tool_block=tool_block,
                facts=facts,
                history=history,
                llm=client,
                desk=desk,
            )
            call = _parse_tool_call(raw)
            if not call:
                break
            name, args = call
            extra = run_tool(name, packet, args)
            label = name if not args else "%s(%s)" % (name, args)
            tools_used.append(label)
            tool_block = (
                f"{tool_block}\n\n### TOOL {label}\n"
                f"{json.dumps(extra, default=str, ensure_ascii=False, indent=2)}"
            )
        else:
            raw = _synthesize(
                question=text,
                tool_block=tool_block + "\n\n(No more tools — answer now.)",
                facts=facts,
                history=history,
                llm=client,
                desk=desk,
            )
    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            "research_llm_failed",
            level=logging.WARNING,
            step="research_chat",
            status="error",
            data={"error": str(exc), "ticker": facts.ticker},
        )
        reply = deterministic_research_answer(packet, question=text)
        return {
            "reply": reply,
            "out_of_scope": False,
            "mode": "research",
            "tools_used": tools_used,
            "repaired": True,
        }

    cleaned = ensure_next_chips(
        polish_research_markdown(
            strip_ungrounded_catalyst_sentences(
                strip_brochure_voice(
                    strip_tool_name_leaks(_normalize_structure(raw))
                ),
                packet,
            )
        )
    )
    if cleaned.lower().startswith("out of scope"):
        return {
            "reply": OUT_OF_SCOPE,
            "out_of_scope": True,
            "mode": "research",
            "tools_used": tools_used,
            "repaired": False,
        }

    check = validate_research_reply(cleaned, facts)
    repaired = False
    reply = cleaned
    if not check.ok:
        repaired_text = repair_research_reply(cleaned, facts, packet=packet)
        recheck = validate_research_reply(repaired_text, facts)
        if recheck.ok and len(repaired_text) >= 40:
            reply = repaired_text
            repaired = True
        else:
            reply = deterministic_research_answer(packet, question=text)
            repaired = True
            check = validate_research_reply(reply, facts)
    else:
        polished = ensure_next_chips(
            polish_research_markdown(
                strip_ungrounded_catalyst_sentences(
                    strip_brochure_voice(strip_tool_name_leaks(cleaned)),
                    packet,
                )
            )
        )
        if polished != cleaned:
            reply = polished
            repaired = True

    log_event(
        logger,
        "research_chat_done",
        step="research_chat",
        status="ok",
        data={
            "ticker": facts.ticker,
            "tools_used": tools_used,
            "repaired": repaired,
            "errors": list(check.errors) if not check.ok else [],
            "targets": targets,
            "desk_tickers": desk.tickers,
        },
    )
    return {
        "reply": reply,
        "out_of_scope": False,
        "mode": "research",
        "tools_used": tools_used,
        "repaired": repaired,
    }


def resolve_chat_mode(
    requested: str | None,
    run_context: dict[str, Any] | None,
) -> str:
    """Return 'guide' or 'research'."""
    mode = (requested or "auto").strip().lower()
    has_run = normalize_desk_context(run_context) is not None
    if mode == "research":
        return "research"
    if mode == "guide":
        return "guide"
    # auto
    return "research" if has_run else "guide"
