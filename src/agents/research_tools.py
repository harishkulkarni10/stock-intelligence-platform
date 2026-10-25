"""Deterministic packet tools for Research Co-Pilot (Phase 1: tools 1–7)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.agents.performance_guardrails import format_projected_move, projected_move_pct

DESK_PACKET_MAX = 20
_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z.&-]{2,}")
_NAME_STOP = frozenset(
    {
        "inc",
        "corp",
        "corporation",
        "ltd",
        "llc",
        "co",
        "the",
        "and",
        "company",
        "group",
        "holdings",
        "plc",
        "sa",
        "ag",
        "nv",
    }
)
# Common spoken names → ticker (used only when that ticker is on the desk).
_BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "KO": ("coke", "coca cola", "coca-cola", "cocacola"),
    "WMT": ("walmart", "wal mart", "wal-mart"),
    "JPM": ("jp morgan", "jpmorgan", "j p morgan", "chase"),
    "ORCL": ("oracle",),
    "AMZN": ("amazon",),
    "PLTR": ("palantir",),
    "MU": ("micron",),
    "AAPL": ("apple",),
    "MSFT": ("microsoft",),
    "GOOGL": ("google", "alphabet"),
    "GOOG": ("google", "alphabet"),
    "META": ("facebook", "meta platforms"),
    "TSLA": ("tesla",),
    "NVDA": ("nvidia",),
    "NFLX": ("netflix",),
    "DIS": ("disney",),
    "BA": ("boeing",),
    "IBM": ("ibm", "international business machines"),
    "AMD": ("amd", "advanced micro devices"),
    "INTC": ("intel",),
    "CRM": ("salesforce",),
    "BLK": ("blackrock",),
}

TOOL_NAMES = (
    "get_run_overview",
    "explain_confidence",
    "get_agent_note",
    "get_forecast_path",
    "get_sources",
    "get_financial_snapshot",
    "get_risk_metrics",
    "compare_labels",
)

AGENT_KEYS = {
    "performance": ("performance_analysis", "performance_trend", "Trend"),
    "news": ("news_summary", "news_sentiment", "Sentiment"),
    "financial": ("financial_analysis", "financial_health", "Health"),
    "risk": ("risk_analysis", "risk_level", "Risk"),
    "report": ("final_report", "recommendation", "Stance"),
}


def truncate_run_context(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact AnalyzeResponse for chat requests (labels, notes, key metrics)."""
    if not payload or not isinstance(payload, dict):
        return None
    ticker = str(payload.get("ticker") or "").strip().upper()
    if not ticker:
        return None

    preds = payload.get("predictions") or {}
    if not isinstance(preds, dict):
        preds = {}
    news = payload.get("news") or {}
    if not isinstance(news, dict):
        news = {}
    financials = payload.get("financials") or {}
    if not isinstance(financials, dict):
        financials = {}
    risk = payload.get("risk") or {}
    if not isinstance(risk, dict):
        risk = {}
    company = payload.get("company") or {}
    if not isinstance(company, dict):
        company = {}

    articles = []
    for item in (news.get("articles") or [])[:8]:
        if not isinstance(item, dict):
            continue
        articles.append(
            {
                "title": item.get("title") or item.get("headline") or "",
                "url": item.get("url") or item.get("link") or "",
                "published": item.get("published")
                or item.get("datetime")
                or item.get("date")
                or "",
            }
        )

    # AnalyzeResponse uses predictions.forecast; some clients send .predictions.
    raw_points = preds.get("forecast") or preds.get("predictions") or []
    pred_points = []
    for item in raw_points[:10]:
        if isinstance(item, dict):
            pred_points.append(
                {
                    "step": item.get("step"),
                    "value": item.get("value"),
                    "date": item.get("date") or item.get("timestamp"),
                }
            )

    def _clip(text: Any, n: int = 1200) -> str:
        s = str(text or "").strip()
        return s if len(s) <= n else s[: n - 1] + "…"

    return {
        "ticker": ticker,
        "status": payload.get("status"),
        "performance_trend": payload.get("performance_trend"),
        "performance_analysis": _clip(payload.get("performance_analysis")),
        "news_sentiment": payload.get("news_sentiment"),
        "news_summary": _clip(payload.get("news_summary")),
        "financial_health": payload.get("financial_health"),
        "financial_analysis": _clip(payload.get("financial_analysis")),
        "risk_level": payload.get("risk_level"),
        "risk_analysis": _clip(payload.get("risk_analysis")),
        "final_report": _clip(payload.get("final_report"), 2000),
        "recommendation": payload.get("recommendation"),
        "confidence": payload.get("confidence"),
        "performance_repaired": bool(payload.get("performance_repaired")),
        "predictions": {
            "last_close": preds.get("last_close"),
            "model_source": preds.get("model_source"),
            "horizon": preds.get("horizon"),
            "predictions": pred_points,
        },
        "financials": {
            "metrics": financials.get("metrics") or {},
            "signals": financials.get("signals") or {},
            "coverage": financials.get("coverage"),
        },
        "risk": {"metrics": risk.get("metrics") or {}},
        "news": {"articles": articles},
        "company": {
            "name": company.get("name") or company.get("longName"),
            "sector": company.get("sector"),
            "industry": company.get("industry"),
        },
        "cached": bool(payload.get("cached")),
        "cache_age_seconds": payload.get("cache_age_seconds"),
        # Legacy hint list; prefer DeskContext.desk_packets for multi-run Research.
        "desk_tickers": [
            str(t).strip().upper()
            for t in (payload.get("desk_tickers") or [])
            if str(t).strip()
        ][:DESK_PACKET_MAX],
    }


@dataclass(frozen=True)
class DeskContext:
    """All finished packets on the desk for this browser visit."""

    active_ticker: str
    packets: dict[str, dict[str, Any]]  # ticker -> truncated packet

    @property
    def tickers(self) -> list[str]:
        return list(self.packets.keys())

    def get(self, ticker: str | None) -> dict[str, Any] | None:
        key = str(ticker or "").strip().upper()
        return self.packets.get(key)


def _ingest_packet(
    store: dict[str, dict[str, Any]],
    raw: dict[str, Any] | None,
) -> str | None:
    packet = truncate_run_context(raw)
    if not packet:
        return None
    ticker = str(packet.get("ticker") or "").upper()
    # Avoid nested desk_tickers noise inside each packet copy.
    packet = {**packet, "desk_tickers": []}
    store[ticker] = packet
    return ticker


def normalize_desk_context(
    run_context: dict[str, Any] | None,
) -> DeskContext | None:
    """Accept legacy single packet or `{active_ticker, desk_packets}` wrapper."""
    if not run_context or not isinstance(run_context, dict):
        return None

    packets: dict[str, dict[str, Any]] = {}
    raw_desk = run_context.get("desk_packets")

    if isinstance(raw_desk, dict):
        for raw in raw_desk.values():
            if isinstance(raw, dict):
                _ingest_packet(packets, raw)
    elif isinstance(raw_desk, list):
        for raw in raw_desk:
            if isinstance(raw, dict):
                _ingest_packet(packets, raw)

    # Legacy: top-level is itself one analyze packet.
    if not packets and run_context.get("ticker"):
        _ingest_packet(packets, run_context)

    if not packets:
        return None

    # Preserve a stable order hint when client sent desk_tickers.
    ordered: dict[str, dict[str, Any]] = {}
    for sym in run_context.get("desk_tickers") or []:
        key = str(sym or "").strip().upper()
        if key in packets and key not in ordered:
            ordered[key] = packets[key]
    for key, value in packets.items():
        if key not in ordered:
            ordered[key] = value
    # Cap payload size for chat.
    if len(ordered) > DESK_PACKET_MAX:
        ordered = dict(list(ordered.items())[:DESK_PACKET_MAX])

    active = str(run_context.get("active_ticker") or "").strip().upper()
    if active not in ordered:
        # Prefer ticker field on wrapper/legacy payload, else first desk entry.
        fallback = str(run_context.get("ticker") or "").strip().upper()
        active = fallback if fallback in ordered else next(iter(ordered))

    # Stamp desk ticker list onto each packet for meta replies / tools.
    desk_list = list(ordered.keys())
    stamped = {
        t: {**pkt, "desk_tickers": desk_list} for t, pkt in ordered.items()
    }
    return DeskContext(active_ticker=active, packets=stamped)


def _normalize_phrase(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _company_name_words(name: str) -> list[str]:
    words = [
        w
        for w in _normalize_phrase(name).split()
        if w and w not in _NAME_STOP and len(w) >= 3
    ]
    return words


def _company_name_hits(question: str, packet: dict[str, Any], ticker: str = "") -> bool:
    """Match company / brand phrasing to a desk packet (Coca Cola → KO)."""
    q_norm = _normalize_phrase(question)
    if not q_norm:
        return False
    q_compact = q_norm.replace(" ", "")

    # Spoken brand aliases when that ticker is on the desk.
    for alias in _BRAND_ALIASES.get(str(ticker or "").upper(), ()):
        alias_norm = _normalize_phrase(alias)
        if not alias_norm:
            continue
        if alias_norm in q_norm or alias_norm.replace(" ", "") in q_compact:
            return True

    name = str((packet.get("company") or {}).get("name") or "").strip()
    words = _company_name_words(name)
    if not words:
        return False

    # Prefer multi-word brand spans: "coca cola" matches "CoCa Cola" / "Coca-Cola".
    for n in range(len(words), 0, -1):
        for i in range(len(words) - n + 1):
            span = words[i : i + n]
            phrase = " ".join(span)
            compact = "".join(span)
            if n == 1 and len(phrase) < 4:
                continue
            if phrase in q_norm or compact in q_compact:
                return True
            # Two+ distinctive tokens anywhere in the question (order-flexible).
            if n >= 2 and all(
                re.search(rf"\b{re.escape(w)}\b", q_norm) for w in span
            ):
                return True
    return False


def mentioned_desk_tickers(question: str, desk: DeskContext) -> list[str]:
    """Desk tickers explicitly referenced by symbol, company name, or brand."""
    q = question or ""
    found: list[str] = []
    for ticker, packet in desk.packets.items():
        hit = bool(re.search(rf"\b{re.escape(ticker)}\b", q, re.IGNORECASE))
        if not hit:
            hit = _company_name_hits(q, packet, ticker)
        if hit and ticker not in found:
            found.append(ticker)
    return found


def resolve_target_tickers(question: str, desk: DeskContext) -> list[str]:
    """Named desk ticker(s) if present; otherwise the active/on-screen packet."""
    named = mentioned_desk_tickers(question, desk)
    if named:
        return named
    active = desk.get(desk.active_ticker)
    if active:
        return [desk.active_ticker]
    return list(desk.packets.keys())[:1]


def off_desk_tickers_requested(question: str, desk: DeskContext) -> list[str]:
    """Tickers/brands the user asked about that are not in finished desk packets."""
    q = question or ""
    q_norm = _normalize_phrase(q)
    q_compact = q_norm.replace(" ", "")
    on_desk = set(desk.packets.keys())
    found: list[str] = []

    for ticker, aliases in _BRAND_ALIASES.items():
        if ticker in on_desk:
            continue
        for alias in aliases:
            alias_norm = _normalize_phrase(alias)
            if not alias_norm:
                continue
            if alias_norm in q_norm or alias_norm.replace(" ", "") in q_compact:
                if ticker not in found:
                    found.append(ticker)
                break
        # Bare symbol mention (e.g. TSLA, AMD) when that ticker is not on desk.
        if ticker not in found and re.search(rf"\b{re.escape(ticker)}\b", q, re.I):
            found.append(ticker)

    return found


def is_multi_ticker_question(question: str, named: list[str]) -> bool:
    """True when the user is clearly asking across more than one desk name."""
    if len(named) >= 2:
        return True
    q = (question or "").lower()
    return bool(
        len(named) >= 1
        and re.search(r"\b(compare|versus|\bvs\b|across|between)\b", q)
    )


def _as_json(data: Any) -> str:
    return json.dumps(data, default=str, ensure_ascii=False, indent=2)


def _forecast_blob(packet: dict[str, Any]) -> dict[str, Any]:
    """Normalize AnalyzeResponse.predictions for projected_move_pct / path tools."""
    preds = packet.get("predictions") or {}
    if not isinstance(preds, dict):
        return {}
    points = preds.get("predictions") or preds.get("forecast") or []
    return {
        "last_close": preds.get("last_close"),
        "last_date": preds.get("last_date"),
        "horizon": preds.get("horizon"),
        "model_source": preds.get("model_source"),
        "predictions": points if isinstance(points, list) else [],
    }


def explain_confidence(packet: dict[str, Any]) -> dict[str, Any]:
    """Desk rules for High / Medium / Low — not 'news alone'."""
    forecast = _forecast_blob(packet)
    source = str(forecast.get("model_source") or "").lower() or "unknown"
    confidence = str(packet.get("confidence") or "—")
    repaired = bool(packet.get("performance_repaired"))
    comparison = compare_labels(packet)
    agreement = comparison.get("agreement") or "partial"

    drivers: list[str] = []
    if source == "persistence" or repaired:
        drivers.append(
            "Forecast trust is limited (persistence baseline or repaired Performance note) "
            "→ desk defaults to Low confidence."
        )
    elif source in {"child", "parent"}:
        drivers.append(
            f"Forecast path comes from the {source} model (not a plain persistence copy)."
        )
    else:
        drivers.append(f"Forecast model_source on this run: {source}.")

    if agreement == "disagreement":
        drivers.append(
            "Specialists disagree (trend vs news and/or financials/risk) — "
            "treat the stance as a working view, not a settled one."
        )
    elif agreement == "aligned":
        drivers.append("Specialist labels lean the same way on this run.")
    else:
        drivers.append("Specialist labels only partly align.")

    if confidence.lower() == "low":
        meaning = (
            "Low means trust the packet lightly — often a simpler forecast path "
            "or a repaired Performance note."
        )
    elif confidence.lower() == "high":
        meaning = (
            "High means the desk sees a stronger forecast-trust setup with less "
            "need for caution flags."
        )
    else:
        meaning = (
            "Medium is the usual rating when a real model path is in play. "
            "It is not 'MIXED news by itself.' News disagreement is a reading tip, "
            "not the sole confidence driver."
        )

    return {
        "confidence": confidence,
        "model_source": source,
        "performance_repaired": repaired,
        "agreement": agreement,
        "meaning": meaning,
        "drivers": drivers,
    }


def get_run_overview(packet: dict[str, Any]) -> dict[str, Any]:
    forecast = _forecast_blob(packet)
    move = projected_move_pct(forecast)
    company = packet.get("company") or {}
    confidence_meta = explain_confidence(packet)
    return {
        "ticker": packet.get("ticker"),
        "company": (company.get("name") if isinstance(company, dict) else None),
        "performance_trend": packet.get("performance_trend"),
        "news_sentiment": packet.get("news_sentiment"),
        "financial_health": packet.get("financial_health"),
        "risk_level": packet.get("risk_level"),
        "stance": packet.get("recommendation"),
        "confidence": packet.get("confidence"),
        "model_source": forecast.get("model_source"),
        "projected_move": format_projected_move(move),
        "projected_move_pct": move,
        "confidence_meaning": confidence_meta.get("meaning"),
        "confidence_drivers": confidence_meta.get("drivers"),
        "agreement": confidence_meta.get("agreement"),
        "cached": packet.get("cached"),
        "cache_age_seconds": packet.get("cache_age_seconds"),
    }


def get_agent_note(packet: dict[str, Any], agent: str) -> dict[str, Any]:
    key = (agent or "").strip().lower()
    if key not in AGENT_KEYS:
        return {
            "error": f"unknown_agent:{agent}",
            "allowed": list(AGENT_KEYS),
        }
    note_key, label_key, label_name = AGENT_KEYS[key]
    return {
        "agent": key,
        "label_name": label_name,
        "label": packet.get(label_key),
        "note": packet.get(note_key) or "",
    }


def get_forecast_path(packet: dict[str, Any]) -> dict[str, Any]:
    forecast = _forecast_blob(packet)
    move = projected_move_pct(forecast)
    return {
        "ticker": packet.get("ticker"),
        "last_close": forecast.get("last_close"),
        "horizon": forecast.get("horizon"),
        "model_source": forecast.get("model_source"),
        "projected_move": format_projected_move(move),
        "projected_move_pct": move,
        "session_values": forecast.get("predictions") or [],
    }


def get_sources(packet: dict[str, Any]) -> dict[str, Any]:
    news = packet.get("news") or {}
    articles = news.get("articles") if isinstance(news, dict) else []
    rows = []
    for item in articles or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "published": item.get("published") or "",
            }
        )
    return {"count": len(rows), "articles": rows}


def get_financial_snapshot(packet: dict[str, Any]) -> dict[str, Any]:
    financials = packet.get("financials") or {}
    if not isinstance(financials, dict):
        financials = {}
    return {
        "health": packet.get("financial_health"),
        "metrics": financials.get("metrics") or {},
        "signals": financials.get("signals") or {},
        "coverage": financials.get("coverage"),
        "source": "this_run_packet",
    }


def get_risk_metrics(packet: dict[str, Any]) -> dict[str, Any]:
    risk = packet.get("risk") or {}
    if not isinstance(risk, dict):
        risk = {}
    return {
        "risk_level": packet.get("risk_level"),
        "metrics": risk.get("metrics") or {},
        "source": "this_run_packet",
    }


def compare_labels(packet: dict[str, Any]) -> dict[str, Any]:
    trend = str(packet.get("performance_trend") or "—").upper()
    news = str(packet.get("news_sentiment") or "—").upper()
    health = str(packet.get("financial_health") or "—").upper()
    risk = str(packet.get("risk_level") or "—").upper()
    stance = str(packet.get("recommendation") or "—").upper()
    confidence = packet.get("confidence") or "—"

    bullish_side = {"BULLISH", "POSITIVE", "STRONG", "CONTAINED"}
    bearish_side = {"BEARISH", "NEGATIVE", "STRESSED", "ELEVATED"}

    def side(label: str) -> str:
        if label in bullish_side:
            return "constructive"
        if label in bearish_side:
            return "cautious"
        return "mixed"

    rows = [
        {"dimension": "Performance trend", "label": trend, "lean": side(trend)},
        {"dimension": "News sentiment", "label": news, "lean": side(news)},
        {"dimension": "Financial health", "label": health, "lean": side(health)},
        {"dimension": "Risk level", "label": risk, "lean": side(risk)},
        {"dimension": "Research stance", "label": stance, "lean": side(stance)},
    ]
    leans = {r["lean"] for r in rows if r["label"] != "—"}
    if len(leans) <= 1:
        agreement = "aligned"
    elif "constructive" in leans and "cautious" in leans:
        agreement = "disagreement"
    elif leans == {"constructive", "mixed"} or leans == {"cautious", "mixed"}:
        # e.g. POSITIVE news vs NEUTRAL/SIDEWAYS path — still a useful tension.
        agreement = "disagreement"
    else:
        agreement = "partial"

    return {
        "ticker": packet.get("ticker"),
        "confidence": confidence,
        "agreement": agreement,
        "rows": rows,
        "note": (
            "Stance follows the performance trend on this desk; news/financials/risk "
            "can disagree and that tension is useful research signal."
        ),
    }


def run_tool(
    name: str,
    packet: dict[str, Any],
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = args or {}
    key = (name or "").strip()
    if key == "get_run_overview":
        return get_run_overview(packet)
    if key == "explain_confidence":
        return explain_confidence(packet)
    if key == "get_agent_note":
        return get_agent_note(packet, str(args.get("agent") or "performance"))
    if key == "get_forecast_path":
        return get_forecast_path(packet)
    if key == "get_sources":
        return get_sources(packet)
    if key == "get_financial_snapshot":
        return get_financial_snapshot(packet)
    if key == "get_risk_metrics":
        return get_risk_metrics(packet)
    if key == "compare_labels":
        return compare_labels(packet)
    return {"error": f"unknown_tool:{name}", "allowed": list(TOOL_NAMES)}


def select_tools_for_question(question: str) -> list[tuple[str, dict[str, Any]]]:
    """Heuristic tool plan so grounding works even without LLM tool-calling."""
    q = (question or "").lower()
    plan: list[tuple[str, dict[str, Any]]] = [("get_run_overview", {})]

    def add(name: str, args: dict[str, Any] | None = None) -> None:
        item = (name, args or {})
        if item not in plan and not any(p[0] == name and p[1] == item[1] for p in plan):
            plan.append(item)

    if any(
        w in q
        for w in (
            "disagree",
            "disagreement",
            "conflict",
            "vs",
            "versus",
            "align",
            "compare",
            "why neutral",
            "why bullish",
            "why bearish",
            "tension",
        )
    ):
        add("compare_labels")

    if any(w in q for w in ("forecast", "projected", "move", "path", "session", "chart")):
        add("get_forecast_path")

    if any(w in q for w in ("source", "headline", "article", "news")):
        add("get_sources")
        add("get_agent_note", {"agent": "news"})

    if any(w in q for w in ("financial", "fundament", "margin", "revenue", "balance")):
        add("get_financial_snapshot")
        add("get_agent_note", {"agent": "financial"})

    if any(w in q for w in ("risk", "downside", "volatility", "drawdown")):
        add("get_risk_metrics")
        add("get_agent_note", {"agent": "risk"})

    if any(w in q for w in ("performance", "trend")):
        add("get_agent_note", {"agent": "performance"})
        add("get_forecast_path")

    if any(
        w in q
        for w in (
            "confidence",
            "how sure",
            "certainty",
            "trust this",
            "why medium",
            "why low",
            "why high",
        )
    ):
        add("explain_confidence")
        add("compare_labels")
        add("get_forecast_path")

    if any(
        w in q
        for w in (
            "what does",
            "what do these",
            "mean when",
            "metric",
            "label mean",
            "definitions",
        )
    ):
        add("explain_confidence")
        add("compare_labels")
        add("get_forecast_path")
        add("get_risk_metrics")

    if any(w in q for w in ("brief", "report", "stance", "bull", "bear", "synthesis")):
        add("get_agent_note", {"agent": "report"})
        add("compare_labels")

    if len(plan) == 1:
        add("compare_labels")
        add("get_forecast_path")

    return plan[:8]


def execute_tool_plan(
    packet: dict[str, Any],
    plan: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    question: str = "",
) -> tuple[list[str], str]:
    """Run tools; return (tool_names_used, formatted_block_for_prompt)."""
    steps = plan if plan is not None else select_tools_for_question(question)
    used: list[str] = []
    blocks: list[str] = []
    for name, args in steps:
        result = run_tool(name, packet, args)
        label = name if not args else f"{name}({args})"
        used.append(label)
        blocks.append(f"### TOOL {label}\n{_as_json(result)}")
    return used, "\n\n".join(blocks)
