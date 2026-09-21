"""Deterministic short-horizon risk metrics for the Risk Analyst.

Numbers come from forecast history + analyze context — not from the LLM.
"""

from __future__ import annotations

import math
from typing import Any

RISK_LABELS = ("CONTAINED", "MODERATE", "ELEVATED")


def _closes_from_history(history: list[Any] | None) -> list[float]:
    closes: list[float] = []
    for point in history or []:
        if not isinstance(point, dict):
            continue
        raw = point.get("value", point.get("close", point.get("Close")))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            closes.append(value)
    return closes


def _pct_returns(closes: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev == 0:
            continue
        out.append((closes[i] / prev) - 1.0)
    return out


def realized_vol(closes: list[float], *, window: int = 20) -> float | None:
    """Daily stdev of returns over the last `window` returns; None if too short."""
    rets = _pct_returns(closes)
    if len(rets) < max(5, window // 2):
        return None
    sample = rets[-window:] if len(rets) >= window else rets
    if len(sample) < 2:
        return None
    mean = sum(sample) / len(sample)
    var = sum((r - mean) ** 2 for r in sample) / (len(sample) - 1)
    daily = math.sqrt(var)
    return daily if math.isfinite(daily) else None


def annualize_daily_vol(daily: float | None) -> float | None:
    if daily is None:
        return None
    return daily * math.sqrt(252.0)


def max_drawdown(closes: list[float]) -> float | None:
    """Most negative peak-to-trough move as a fraction (e.g. -0.22)."""
    if len(closes) < 5:
        return None
    peak = closes[0]
    worst = 0.0
    for price in closes:
        if price > peak:
            peak = price
        if peak <= 0:
            continue
        dd = (price / peak) - 1.0
        if dd < worst:
            worst = dd
    return worst if math.isfinite(worst) else None


def projected_move_pct(forecast: dict[str, Any] | None) -> float | None:
    data = forecast or {}
    last_close = data.get("last_close")
    preds = data.get("predictions") or []
    if last_close is None or not preds:
        return None
    try:
        terminal = float(preds[-1].get("value"))
        base = float(last_close)
        if base == 0:
            return None
        return ((terminal / base) - 1.0) * 100.0
    except (TypeError, ValueError, AttributeError, ZeroDivisionError):
        return None


def move_vs_daily_vol(move_pct: float | None, daily_vol: float | None) -> float | None:
    """Absolute projected move (fraction) divided by daily vol — how many sigma."""
    if move_pct is None or daily_vol is None or daily_vol <= 1e-4:
        return None
    return abs(move_pct / 100.0) / daily_vol


def derive_allowed_risk(
    *,
    model_source: str | None,
    performance_repaired: bool,
    vol_ann_20d: float | None,
    max_dd: float | None,
    move_vol_ratio: float | None,
    financial_health: str | None,
    leverage_signal: str | None,
    news_sentiment: str | None,
) -> str:
    """Code-assigned Risk label. Higher score → more elevated."""
    score = 0
    source = (model_source or "").lower()
    if source == "persistence":
        score += 2
    if performance_repaired:
        score += 1

    if vol_ann_20d is not None:
        if vol_ann_20d >= 0.50:
            score += 2
        elif vol_ann_20d >= 0.32:
            score += 1

    if max_dd is not None:
        if max_dd <= -0.30:
            score += 2
        elif max_dd <= -0.18:
            score += 1

    if move_vol_ratio is not None:
        if move_vol_ratio >= 3.0:
            score += 2
        elif move_vol_ratio >= 1.75:
            score += 1

    health = (financial_health or "").upper()
    if health == "STRESSED":
        score += 2
    elif health in {"UNAVAILABLE", ""}:
        score += 1

    lev = (leverage_signal or "").lower()
    if lev == "stretched":
        score += 2
    elif lev == "moderate":
        score += 1

    sentiment = (news_sentiment or "").upper()
    if sentiment == "NEGATIVE":
        score += 2
    elif sentiment in {"MIXED", "UNAVAILABLE"}:
        score += 1

    if score >= 5:
        return "ELEVATED"
    if score <= 1:
        return "CONTAINED"
    return "MODERATE"


def build_risk_metrics(
    *,
    ticker: str,
    forecast: dict[str, Any] | None,
    performance_trend: str | None = None,
    performance_repaired: bool = False,
    news_sentiment: str | None = None,
    financial_health: str | None = None,
    financials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable risk facts object for prompts and guardrails."""
    data = forecast or {}
    closes = _closes_from_history(
        data.get("history") if isinstance(data.get("history"), list) else []
    )
    if len(closes) > 260:
        closes = closes[-260:]

    daily_20 = realized_vol(closes, window=20)
    daily_60 = realized_vol(closes, window=60)
    vol_ann_20 = annualize_daily_vol(daily_20)
    vol_ann_60 = annualize_daily_vol(daily_60)
    dd = max_drawdown(closes)
    move_pct = projected_move_pct(data)
    mv_ratio = move_vs_daily_vol(move_pct, daily_20)

    fin = financials or {}
    signals = fin.get("signals") if isinstance(fin.get("signals"), dict) else {}
    leverage = str(signals.get("leverage") or "") or None

    allowed = derive_allowed_risk(
        model_source=str(data.get("model_source") or ""),
        performance_repaired=bool(performance_repaired),
        vol_ann_20d=vol_ann_20,
        max_dd=dd,
        move_vol_ratio=mv_ratio,
        financial_health=financial_health,
        leverage_signal=leverage,
        news_sentiment=news_sentiment,
    )

    def _r(value: float | None, ndigits: int = 4) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        return round(float(value), ndigits)

    metrics = {
        "vol_daily_20d": _r(daily_20, 6),
        "vol_ann_20d": _r(vol_ann_20, 4),
        "vol_ann_60d": _r(vol_ann_60, 4),
        "max_drawdown": _r(dd, 4),
        "projected_move_pct": _r(move_pct, 2),
        "move_vs_daily_vol": _r(mv_ratio, 2),
        "history_points": len(closes),
    }

    allowed_numbers: list[float] = []
    for key in (
        "vol_daily_20d",
        "vol_ann_20d",
        "vol_ann_60d",
        "max_drawdown",
        "projected_move_pct",
        "move_vs_daily_vol",
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            allowed_numbers.append(float(value))
            if key.startswith("vol_ann") or key == "max_drawdown":
                allowed_numbers.append(round(abs(float(value)) * 100.0, 2))

    drivers: list[str] = []
    source = str(data.get("model_source") or "unknown")
    if source == "persistence":
        drivers.append("forecast_source_persistence")
    if performance_repaired:
        drivers.append("performance_note_repaired")
    if vol_ann_20 is not None and vol_ann_20 >= 0.32:
        drivers.append("elevated_realized_volatility")
    if dd is not None and dd <= -0.18:
        drivers.append("material_historical_drawdown")
    if mv_ratio is not None and mv_ratio >= 1.75:
        drivers.append("large_projected_move_vs_vol")
    if (financial_health or "").upper() == "STRESSED":
        drivers.append("stressed_financial_health")
    if (leverage or "").lower() == "stretched":
        drivers.append("stretched_leverage")
    if (news_sentiment or "").upper() == "NEGATIVE":
        drivers.append("negative_news_tone")
    elif (news_sentiment or "").upper() == "UNAVAILABLE":
        drivers.append("thin_or_missing_news")

    return {
        "status": "ok" if closes or move_pct is not None else "partial",
        "ticker": str(ticker or data.get("ticker") or "").upper(),
        "allowed_risk": allowed,
        "model_source": source,
        "performance_trend": (performance_trend or "").upper() or None,
        "performance_repaired": bool(performance_repaired),
        "news_sentiment": (news_sentiment or "").upper() or None,
        "financial_health": (financial_health or "").upper() or None,
        "leverage_signal": leverage,
        "metrics": metrics,
        "drivers": drivers,
        "allowed_numbers": allowed_numbers,
    }


def format_risk_metrics_for_prompt(risk: dict[str, Any] | None) -> str:
    data = risk or {}
    m = data.get("metrics") or {}

    def pct_frac(key: str) -> str:
        value = m.get(key)
        if value is None:
            return "n/a"
        return f"{float(value) * 100:.1f}%"

    def num(key: str, suffix: str = "") -> str:
        value = m.get(key)
        if value is None:
            return "n/a"
        return f"{float(value):.2f}{suffix}"

    drivers = data.get("drivers") or []
    driver_line = ", ".join(drivers) if drivers else "(none flagged)"
    return (
        f"Ticker: {data.get('ticker')}\n"
        f"Allowed risk label (code): {data.get('allowed_risk')}\n"
        f"Model source: {data.get('model_source')}\n"
        f"Performance trend: {data.get('performance_trend') or 'n/a'} "
        f"(repaired={data.get('performance_repaired')})\n"
        f"News sentiment: {data.get('news_sentiment') or 'n/a'}\n"
        f"Financial health: {data.get('financial_health') or 'n/a'} "
        f"(leverage={data.get('leverage_signal') or 'n/a'})\n"
        f"History points used: {m.get('history_points', 0)}\n"
        f"Realized vol (ann., ~20d): {pct_frac('vol_ann_20d')}\n"
        f"Realized vol (ann., ~60d): {pct_frac('vol_ann_60d')}\n"
        f"Max drawdown (history window): {pct_frac('max_drawdown')}\n"
        f"Projected move: {num('projected_move_pct', '%')}\n"
        f"Projected move vs daily vol (sigma): {num('move_vs_daily_vol')}\n"
        f"Flagged drivers: {driver_line}\n"
    )
