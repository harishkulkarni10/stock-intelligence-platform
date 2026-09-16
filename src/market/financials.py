"""Deterministic fundamental snapshot for the Financial Analyst (agent 3).

Numbers come from Yahoo / yfinance. The LLM only interprets this payload.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

from logger.logger import get_logger, log_event
from src.data.ingestion import normalize_ticker

logger = get_logger("sip.market.financials")

HEALTH_LABELS = ("STRONG", "ADEQUATE", "STRESSED", "UNAVAILABLE")


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _pct_points(value: Any) -> float | None:
    """Normalize Yahoo growth/margin fields to percent points (e.g. 0.12 → 12.0)."""
    number = _f(value)
    if number is None:
        return None
    if abs(number) <= 1.5:
        return number * 100.0
    return number


def _row_pair(frame: Any, labels: tuple[str, ...]) -> tuple[float | None, float | None]:
    if frame is None:
        return None, None
    try:
        if getattr(frame, "empty", True):
            return None, None
        index_map = {str(idx).strip().lower(): idx for idx in frame.index}
        for label in labels:
            key = label.lower()
            if key in index_map:
                idx = index_map[key]
                cols = list(frame.columns)
                latest = _f(frame.loc[idx, cols[0]]) if cols else None
                prior = _f(frame.loc[idx, cols[1]]) if len(cols) > 1 else None
                return latest, prior
    except Exception:  # noqa: BLE001
        return None, None
    return None, None


def _yoy_pct(latest: float | None, prior: float | None) -> float | None:
    if latest is None or prior is None or prior == 0:
        return None
    return ((latest / prior) - 1.0) * 100.0


def _signal_profitability(
    *,
    net_margin_pct: float | None,
    op_margin_pct: float | None,
    fcf: float | None,
) -> str:
    margin = net_margin_pct if net_margin_pct is not None else op_margin_pct
    if margin is None and fcf is None:
        return "unknown"
    if (margin is not None and margin < 0) or (fcf is not None and fcf < 0):
        return "weak"
    if margin is not None and margin >= 15:
        return "improving" if (fcf is None or fcf >= 0) else "stable"
    if margin is not None and margin >= 5:
        return "stable"
    if margin is not None:
        return "weak"
    return "stable" if fcf is not None and fcf >= 0 else "unknown"


def _signal_leverage(
    *,
    debt_to_equity: float | None,
    total_debt: float | None,
    total_cash: float | None,
    current_ratio: float | None,
) -> str:
    if (
        debt_to_equity is None
        and total_debt is None
        and total_cash is None
        and current_ratio is None
    ):
        return "unknown"
    if debt_to_equity is not None:
        # Yahoo debtToEquity is often already scaled (e.g. 45.2 = 45.2%).
        de = debt_to_equity / 100.0 if debt_to_equity > 5 else debt_to_equity
        if de >= 1.5:
            return "stretched"
        if de >= 0.8:
            return "moderate"
        return "conservative"
    if total_debt is not None and total_cash is not None:
        net = total_debt - total_cash
        if total_debt > 0 and net > total_debt * 0.7:
            return "stretched"
        if net <= 0:
            return "conservative"
        return "moderate"
    if current_ratio is not None and current_ratio < 1.0:
        return "stretched"
    return "moderate"


def _signal_valuation(
    *,
    pe: float | None,
    ps: float | None,
    revenue_yoy_pct: float | None,
) -> str:
    if pe is None and ps is None:
        return "unknown"
    # Heuristic bands only — disclosed as such to the LLM.
    if pe is not None:
        if pe < 0:
            return "unknown"
        if pe >= 45:
            return "rich"
        if pe <= 15:
            return "cheap"
        return "fair"
    if ps is not None:
        growth = revenue_yoy_pct or 0.0
        if ps >= 12 and growth < 20:
            return "rich"
        if ps <= 2:
            return "cheap"
        return "fair"
    return "unknown"


def derive_allowed_health(
    *,
    coverage: str,
    profitability_signal: str,
    leverage_signal: str,
    fcf: float | None,
    net_margin_pct: float | None,
) -> str:
    if coverage == "missing":
        return "UNAVAILABLE"
    stressed = (
        leverage_signal == "stretched"
        or profitability_signal == "weak"
        or (fcf is not None and fcf < 0 and (net_margin_pct is None or net_margin_pct < 5))
    )
    strong = (
        profitability_signal in {"improving", "stable"}
        and leverage_signal in {"conservative", "moderate"}
        and (fcf is None or fcf >= 0)
        and (net_margin_pct is None or net_margin_pct >= 8)
        and coverage == "full"
    )
    if stressed:
        return "STRESSED"
    if strong:
        return "STRONG"
    if coverage in {"full", "partial"}:
        return "ADEQUATE"
    return "UNAVAILABLE"


def _collect_allowed_numbers(metrics: dict[str, Any]) -> list[float]:
    allowed: list[float] = []
    for value in metrics.values():
        number = _f(value)
        if number is None:
            continue
        allowed.append(number)
        # Also allow percent-style and scaled variants the model may write.
        allowed.append(round(number, 2))
        allowed.append(round(number, 1))
        if abs(number) >= 1:
            allowed.append(round(number / 1e6, 2))
            allowed.append(round(number / 1e9, 2))
        if abs(number) <= 1000:
            allowed.append(round(number / 100.0, 4))
            allowed.append(round(number * 100.0, 2))
    # Deduplicate while preserving order.
    seen: set[float] = set()
    unique: list[float] = []
    for item in allowed:
        key = round(item, 6)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def get_financials(ticker: str) -> dict[str, Any]:
    """Fetch a normalized fundamental snapshot for agent 3."""
    symbol = normalize_ticker(ticker)
    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}
    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            "financials_failed",
            step="financials",
            status="error",
            data={"ticker": symbol, "error": str(exc)},
        )
        return {
            "status": "error",
            "ticker": symbol,
            "coverage": "missing",
            "allowed_health": "UNAVAILABLE",
            "error": str(exc),
            "metrics": {},
            "signals": {},
            "allowed_numbers": [],
        }

    revenue = _f(info.get("totalRevenue"))
    revenue_growth = _pct_points(info.get("revenueGrowth"))
    gross_margin = _pct_points(info.get("grossMargins"))
    op_margin = _pct_points(info.get("operatingMargins"))
    net_margin = _pct_points(info.get("profitMargins"))
    operating_cf = _f(info.get("operatingCashflow"))
    free_cf = _f(info.get("freeCashflow"))
    total_cash = _f(info.get("totalCash"))
    total_debt = _f(info.get("totalDebt"))
    debt_to_equity = _f(info.get("debtToEquity"))
    current_ratio = _f(info.get("currentRatio"))
    roe = _pct_points(info.get("returnOnEquity"))
    roa = _pct_points(info.get("returnOnAssets"))
    market_cap = _f(info.get("marketCap"))
    pe = _f(info.get("trailingPE")) or _f(info.get("forwardPE"))
    ps = _f(info.get("priceToSalesTrailing12Months"))
    pb = _f(info.get("priceToBook"))
    ev_ebitda = _f(info.get("enterpriseToEbitda"))
    currency = str(info.get("currency") or "USD").strip() or "USD"
    name = str(info.get("shortName") or info.get("longName") or symbol).strip() or symbol
    sector = str(info.get("sector") or "").strip() or None
    industry = str(info.get("industry") or "").strip() or None

    # Prefer statement-derived revenue YoY when info growth is missing.
    revenue_yoy = revenue_growth
    try:
        income = stock.income_stmt
        latest_rev, prior_rev = _row_pair(
            income,
            ("Total Revenue", "Operating Revenue", "Revenue"),
        )
        if revenue is None:
            revenue = latest_rev
        statement_yoy = _yoy_pct(latest_rev, prior_rev)
        if revenue_yoy is None:
            revenue_yoy = statement_yoy
    except Exception:  # noqa: BLE001
        pass

    metrics = {
        "revenue": revenue,
        "revenue_yoy_pct": revenue_yoy,
        "gross_margin_pct": gross_margin,
        "operating_margin_pct": op_margin,
        "net_margin_pct": net_margin,
        "operating_cashflow": operating_cf,
        "free_cashflow": free_cf,
        "total_cash": total_cash,
        "total_debt": total_debt,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "roe_pct": roe,
        "roa_pct": roa,
        "market_cap": market_cap,
        "pe": pe,
        "ps": ps,
        "pb": pb,
        "ev_ebitda": ev_ebitda,
    }
    present = sum(1 for value in metrics.values() if value is not None)
    core = sum(
        1
        for key in ("revenue", "net_margin_pct", "free_cashflow", "total_debt", "pe", "market_cap")
        if metrics.get(key) is not None
    )
    if present == 0:
        coverage = "missing"
        status = "missing"
    elif core >= 4 and present >= 8:
        coverage = "full"
        status = "ok"
    else:
        coverage = "partial"
        status = "partial"

    profitability_signal = _signal_profitability(
        net_margin_pct=net_margin, op_margin_pct=op_margin, fcf=free_cf
    )
    leverage_signal = _signal_leverage(
        debt_to_equity=debt_to_equity,
        total_debt=total_debt,
        total_cash=total_cash,
        current_ratio=current_ratio,
    )
    valuation_signal = _signal_valuation(pe=pe, ps=ps, revenue_yoy_pct=revenue_yoy)
    allowed_health = derive_allowed_health(
        coverage=coverage,
        profitability_signal=profitability_signal,
        leverage_signal=leverage_signal,
        fcf=free_cf,
        net_margin_pct=net_margin,
    )

    payload = {
        "status": status,
        "ticker": symbol,
        "name": name,
        "sector": sector,
        "industry": industry,
        "currency": currency,
        "coverage": coverage,
        "as_of": None,
        "source": "yfinance",
        "metrics": metrics,
        "signals": {
            "profitability": profitability_signal,
            "leverage": leverage_signal,
            "valuation": valuation_signal,
        },
        "allowed_health": allowed_health,
        "allowed_numbers": _collect_allowed_numbers(metrics),
    }
    log_event(
        logger,
        "financials_ok",
        step="financials",
        status=status,
        data={
            "ticker": symbol,
            "coverage": coverage,
            "allowed_health": allowed_health,
            "metrics_present": present,
        },
    )
    return payload


def format_financials_for_prompt(financials: dict[str, Any]) -> str:
    """Compact text block of fundamental facts for the LLM."""
    symbol = financials.get("ticker") or "UNKNOWN"
    if financials.get("status") in {"error", "missing"} or financials.get("coverage") == "missing":
        detail = financials.get("error") or "no usable fundamental fields"
        return (
            f"Fundamentals unavailable for {symbol}: {detail}. "
            "Do not invent revenue, margins, debt, or valuation multiples."
        )

    metrics = financials.get("metrics") or {}
    signals = financials.get("signals") or {}

    def fmt(key: str, *, pct: bool = False, money: bool = False) -> str:
        value = metrics.get(key)
        if value is None:
            return "n/a"
        number = float(value)
        if pct:
            return f"{number:.1f}%"
        if money:
            abs_n = abs(number)
            if abs_n >= 1e9:
                return f"{number / 1e9:.2f}B"
            if abs_n >= 1e6:
                return f"{number / 1e6:.2f}M"
            return f"{number:.2f}"
        return f"{number:.2f}"

    lines = [
        f"Fundamentals for {symbol} ({financials.get('name')})",
        f"Sector/Industry: {financials.get('sector') or 'n/a'} / {financials.get('industry') or 'n/a'}",
        f"Currency: {financials.get('currency')}",
        f"Coverage: {financials.get('coverage')}",
        f"Allowed health label (from code): {financials.get('allowed_health')}",
        f"Signals: profitability={signals.get('profitability')}, "
        f"leverage={signals.get('leverage')}, valuation={signals.get('valuation')} "
        "(valuation bands are heuristics only)",
        "Metrics:",
        f"- revenue: {fmt('revenue', money=True)}",
        f"- revenue_yoy_pct: {fmt('revenue_yoy_pct', pct=True)}",
        f"- gross_margin_pct: {fmt('gross_margin_pct', pct=True)}",
        f"- operating_margin_pct: {fmt('operating_margin_pct', pct=True)}",
        f"- net_margin_pct: {fmt('net_margin_pct', pct=True)}",
        f"- operating_cashflow: {fmt('operating_cashflow', money=True)}",
        f"- free_cashflow: {fmt('free_cashflow', money=True)}",
        f"- total_cash: {fmt('total_cash', money=True)}",
        f"- total_debt: {fmt('total_debt', money=True)}",
        f"- debt_to_equity: {fmt('debt_to_equity')}",
        f"- current_ratio: {fmt('current_ratio')}",
        f"- roe_pct: {fmt('roe_pct', pct=True)}",
        f"- roa_pct: {fmt('roa_pct', pct=True)}",
        f"- market_cap: {fmt('market_cap', money=True)}",
        f"- pe: {fmt('pe')}",
        f"- ps: {fmt('ps')}",
        f"- pb: {fmt('pb')}",
        f"- ev_ebitda: {fmt('ev_ebitda')}",
    ]
    return "\n".join(lines)
