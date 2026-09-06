"""Product UI for the stock intelligence workflow."""

from __future__ import annotations

import html
import os
import re
import uuid
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", os.getenv("API_BASE_URL", "http://localhost:8000")).rstrip("/")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("UI_REQUEST_TIMEOUT_SECONDS", "600"))

st.set_page_config(
    page_title="Stock Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap');
    :root {
        --ink: #e9eef2;
        --muted: #82909c;
        --panel: #111920;
        --line: #24313a;
        --cyan: #53d6c7;
        --amber: #f1b65b;
        --red: #ff7474;
    }
    .stApp {
        background:
            radial-gradient(circle at 88% 0%, rgba(83,214,199,.08), transparent 28rem),
            linear-gradient(180deg, #091015 0%, #0c1319 100%);
        color: var(--ink);
        font-family: "Manrope", sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stDecoration"] { display: none; }
    .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 5rem; }
    h1, h2, h3 { font-family: "Manrope", sans-serif !important; letter-spacing: -.03em; }
    code, .eyebrow, .mono { font-family: "DM Mono", monospace !important; }
    .brandline {
        display: flex; justify-content: space-between; align-items: center;
        border-bottom: 1px solid var(--line); padding-bottom: 1rem; margin-bottom: 2.4rem;
    }
    .mark { color: var(--cyan); font: 500 1rem "DM Mono"; letter-spacing: .12em; }
    .system-id { color: var(--muted); font: .72rem "DM Mono"; letter-spacing: .09em; }
    .eyebrow { color: var(--cyan); font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; }
    .hero-title { font-size: clamp(2.7rem, 6vw, 5.4rem); line-height: .96; max-width: 900px; margin: .7rem 0 1.1rem; }
    .hero-copy { color: #a8b3bc; max-width: 680px; font-size: 1.02rem; line-height: 1.7; }
    .status-strip {
        display: flex; gap: .55rem; flex-wrap: wrap; margin: 1.2rem 0 2rem;
    }
    .status-chip {
        border: 1px solid var(--line); background: rgba(17,25,32,.75);
        border-radius: 99px; padding: .34rem .7rem; color: #9daab4;
        font: .68rem "DM Mono"; text-transform: uppercase; letter-spacing: .05em;
    }
    .status-chip.ok::before { content: "●"; color: var(--cyan); margin-right: .45rem; }
    .status-chip.off::before { content: "○"; color: var(--red); margin-right: .45rem; }
    .status-chip.optional::before { content: "○"; color: var(--amber); margin-right: .45rem; }
    div[data-testid="stForm"] {
        border: 1px solid var(--line); border-radius: 16px; background: rgba(17,25,32,.82);
        padding: 1rem 1.1rem .35rem;
    }
    .stTextInput input {
        font-family: "DM Mono", monospace; font-size: 1.15rem; letter-spacing: .08em;
        background: #0b1217; border-color: #2a3943;
    }
    .stButton button, .stFormSubmitButton button {
        border: 0; border-radius: 9px; background: var(--cyan); color: #07110f;
        font-weight: 700; min-height: 2.85rem;
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        background: #7ce5d9; color: #07110f;
    }
    .decision {
        border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
        padding: 1.6rem 0; margin: 1.4rem 0 2.2rem;
        display: grid; grid-template-columns: 1.1fr .6fr .6fr .7fr; gap: 1rem;
    }
    .decision-cell { border-left: 1px solid var(--line); padding-left: 1rem; }
    .decision-cell:first-child { border-left: 0; padding-left: 0; }
    .metric-label { color: var(--muted); font: .67rem "DM Mono"; letter-spacing: .1em; text-transform: uppercase; }
    .metric-value { color: var(--ink); font-size: 1.55rem; font-weight: 650; margin-top: .32rem; }
    .metric-value.positive { color: var(--cyan); }
    .metric-value.negative { color: var(--red); }
    .section-index { color: var(--cyan); font: .72rem "DM Mono"; margin-bottom: .35rem; }
    .agent-card {
        height: 100%; min-height: 180px; border: 1px solid var(--line); border-radius: 14px;
        background: rgba(17,25,32,.72); padding: 1.2rem 1.25rem;
    }
    .agent-name { font-size: 1rem; font-weight: 650; margin: .25rem 0 .8rem; }
    .agent-copy { color: #b5c0c8; font-size: .9rem; line-height: 1.65; white-space: pre-wrap; }
    .source-card { border-left: 2px solid var(--amber); padding: .2rem 0 .2rem 1rem; margin: .8rem 0; }
    .source-title { color: #dce3e7; font-size: .87rem; font-weight: 600; }
    .source-meta { color: var(--muted); font: .68rem "DM Mono"; margin-top: .25rem; }
    .report-shell {
        border: 1px solid #2b3c46; border-radius: 16px; background: #101820;
        padding: clamp(1.2rem, 3vw, 2.4rem); margin-top: .8rem;
    }
    .disclaimer { color: #687680; font-size: .72rem; margin-top: 3rem; }
    @media (max-width: 760px) {
        .decision { grid-template-columns: 1fr 1fr; }
        .decision-cell:nth-child(3) { border-left: 0; padding-left: 0; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def _readiness() -> tuple[bool, dict[str, bool]]:
    try:
        response = requests.get(f"{API_URL}/ready", timeout=3)
        if response.status_code != 200:
            return False, {}
        payload = response.json()
        return True, payload.get("details", {})
    except (requests.RequestException, ValueError):
        return False, {}


def _status_strip(api_ok: bool, details: dict[str, bool]) -> str:
    entries = [
        ("API", api_ok, False),
        ("Forecast model", details.get("parent_model", False), False),
        ("Feature data", details.get("features", False), False),
        ("Redis cache", details.get("redis", False), True),
    ]
    chips = []
    for label, available, optional in entries:
        kind = "ok" if available else ("optional" if optional else "off")
        suffix = " optional" if optional and not available else ""
        chips.append(f'<span class="status-chip {kind}">{_safe(label + suffix)}</span>')
    return f'<div class="status-strip">{"".join(chips)}</div>'


def _forecast_frame(predictions: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.DataFrame(predictions.get("history") or [])
    forecast = pd.DataFrame(predictions.get("forecast") or [])
    if not history.empty:
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history["value"] = pd.to_numeric(history.get("close"), errors="coerce")
        history = history.dropna(subset=["date", "value"]).sort_values("date")
    if not forecast.empty:
        forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
        source = "value" if "value" in forecast else "close"
        forecast["value"] = pd.to_numeric(forecast.get(source), errors="coerce")
        forecast = forecast.dropna(subset=["date", "value"]).sort_values("date")
    return history, forecast


def _forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, ticker: str) -> go.Figure:
    figure = go.Figure()
    if not history.empty:
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["value"],
                name="Observed",
                mode="lines",
                line={"color": "#82909c", "width": 2},
                hovertemplate="%{x|%d %b}<br>$%{y:,.2f}<extra>Observed</extra>",
            )
        )
    if not forecast.empty:
        x_values = forecast["date"].tolist()
        y_values = forecast["value"].tolist()
        if not history.empty:
            x_values.insert(0, history.iloc[-1]["date"])
            y_values.insert(0, history.iloc[-1]["value"])
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                name="Model forecast",
                mode="lines+markers",
                line={"color": "#53d6c7", "width": 3},
                marker={"size": 7, "color": "#53d6c7"},
                hovertemplate="%{x|%d %b}<br>$%{y:,.2f}<extra>Forecast</extra>",
            )
        )
    figure.update_layout(
        title={"text": f"{ticker} · observed vs model path", "font": {"size": 15}},
        height=390,
        margin={"l": 5, "r": 5, "t": 45, "b": 5},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Mono", "color": "#82909c", "size": 11},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        xaxis={"gridcolor": "#1d2931", "showline": False},
        yaxis={"gridcolor": "#1d2931", "tickprefix": "$", "showline": False},
    )
    return figure


def _agent_card(index: str, name: str, role: str, content: str) -> None:
    st.markdown(
        f"""
        <div class="agent-card">
          <div class="eyebrow">{_safe(index)} · {_safe(role)}</div>
          <div class="agent-name">{_safe(name)}</div>
          <div class="agent-copy">{_safe(content or "No response returned.")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        return str(payload.get("detail") or payload)
    except ValueError:
        return response.text or "Unknown API error"


st.markdown(
    """
    <div class="brandline">
      <div class="mark">SIP / RESEARCH SYSTEM</div>
      <div class="system-id">LOCAL INTELLIGENCE WORKSPACE · V0.1</div>
    </div>
    """,
    unsafe_allow_html=True,
)

api_ok, readiness = _readiness()
st.markdown('<div class="eyebrow">LSTM forecast × performance analyst</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Forecast first.<br>Agent 1 interprets it.</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Production path for now: live champion forecast '
    "(child LSTM, parent, or persistence) plus the Performance Analyst with "
    "guardrails. News, report writer, and critic are parked until agent 1 is solid.</div>",
    unsafe_allow_html=True,
)
st.markdown(_status_strip(api_ok, readiness), unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.form("research_command"):
    input_col, action_col = st.columns([5, 1], vertical_alignment="bottom")
    with input_col:
        ticker_input = st.text_input(
            "RESEARCH SYMBOL",
            value=st.session_state.get("ticker", "NVDA"),
            placeholder="NVDA",
            help="US equity ticker, for example NVDA, AAPL, MSFT, or TSLA.",
        )
    with action_col:
        submitted = st.form_submit_button("Run agent 1 →", use_container_width=True)

if submitted:
    ticker = re.sub(r"[^A-Za-z0-9.^=-]", "", ticker_input.strip()).upper()
    if not ticker:
        st.error("Enter a valid ticker.")
    elif not api_ok:
        st.error(f"The API is not reachable at {API_URL}. Start it, then retry.")
    else:
        st.session_state.ticker = ticker
        try:
            with st.status(f"Running agent 1 for {ticker}…", expanded=True) as status:
                st.write("Loading champion forecast (LSTM / persistence)")
                st.write("Performance Analyst + guardrails (Ollama)")
                response = requests.post(
                    f"{API_URL}/analyze",
                    json={"ticker": ticker, "thread_id": st.session_state.thread_id},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code != 200:
                    status.update(label="Agent 1 workflow failed", state="error")
                    st.session_state.analysis_error = _error_detail(response)
                    st.session_state.pop("analysis", None)
                else:
                    payload = response.json()
                    if payload.get("status") != "ok":
                        status.update(label="Forecast / agent unavailable", state="error")
                        st.session_state.analysis_error = payload.get("detail") or payload.get("status")
                        st.session_state.pop("analysis", None)
                    else:
                        st.session_state.analysis = payload
                        st.session_state.pop("analysis_error", None)
                        cache_note = " · cache hit" if payload.get("cached") else ""
                        status.update(label=f"{ticker} agent 1 ready{cache_note}", state="complete")
        except requests.Timeout:
            st.session_state.analysis_error = (
                f"The analysis exceeded {REQUEST_TIMEOUT_SECONDS} seconds. "
                "Ollama or market-data retrieval may still be running."
            )
            st.session_state.pop("analysis", None)
        except (requests.RequestException, ValueError) as exc:
            st.session_state.analysis_error = f"Could not complete the API request: {exc}"
            st.session_state.pop("analysis", None)

if error := st.session_state.get("analysis_error"):
    st.error(error)

data = st.session_state.get("analysis")
if data:
    ticker = data.get("ticker", st.session_state.get("ticker", ""))
    predictions = data.get("predictions") or {}
    history, forecast = _forecast_frame(predictions)
    last_close = predictions.get("last_close")
    terminal_value = float(forecast.iloc[-1]["value"]) if not forecast.empty else None
    projected_change = (
        ((terminal_value / float(last_close)) - 1) * 100
        if terminal_value is not None and last_close not in (None, 0)
        else None
    )
    recommendation = str(data.get("recommendation") or "NEUTRAL").upper()
    trend = str(data.get("performance_trend") or recommendation).upper()
    stance_class = "positive" if trend == "BULLISH" else (
        "negative" if trend == "BEARISH" else ""
    )
    guardrail = data.get("performance_guardrail_ok")
    guardrail_label = "PASS" if guardrail is True else ("FAIL" if guardrail is False else "—")

    st.markdown(
        f"""
        <div class="decision">
          <div class="decision-cell">
            <div class="metric-label">Forecast trend</div>
            <div class="metric-value {stance_class}">{_safe(trend)}</div>
          </div>
          <div class="decision-cell">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{_safe(data.get("confidence"))}</div>
          </div>
          <div class="decision-cell">
            <div class="metric-label">Last close</div>
            <div class="metric-value">{f"${float(last_close):,.2f}" if last_close is not None else "—"}</div>
          </div>
          <div class="decision-cell">
            <div class="metric-label">Guardrails</div>
            <div class="metric-value">{_safe(guardrail_label)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-index">01 / MODEL FORECAST</div>', unsafe_allow_html=True)
    chart_col, model_col = st.columns([2.2, 1])
    with chart_col:
        st.plotly_chart(
            _forecast_chart(history, forecast, ticker),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with model_col:
        st.markdown("### Forecast provenance")
        st.metric("Model path", str(predictions.get("model_source") or "—").upper())
        st.metric("Forecast horizon", f"{predictions.get('horizon') or len(forecast)} sessions")
        st.metric("Projected move", f"{projected_change:+.2f}%" if projected_change is not None else "—")
        st.caption(f"Version · {predictions.get('model_version') or 'unversioned'}")
        st.caption(f"Last observation · {predictions.get('last_date') or '—'}")
        if data.get("performance_repaired"):
            st.caption("Analysis was repaired by guardrails after an LLM mismatch.")
        if not forecast.empty:
            with st.expander("Forecast values"):
                display = forecast[["date", "value"]].copy()
                display["date"] = display["date"].dt.strftime("%Y-%m-%d")
                display["value"] = display["value"].map(lambda value: f"${value:,.2f}")
                st.dataframe(display, hide_index=True, use_container_width=True)

    st.markdown('<div class="section-index">02 / AGENT 1 · PERFORMANCE ANALYST</div>', unsafe_allow_html=True)
    _agent_card(
        "A1",
        "Performance analyst",
        "Interprets the champion forecast only (guardrailed)",
        data.get("performance_analysis") or "",
    )
    st.caption(
        f"Mode · {data.get('mode') or 'performance_only'} · "
        f"{'served from Redis cache' if data.get('cached') else 'fresh run'} · "
        "Agents 2–4 disabled in this release"
    )

st.markdown(
    '<div class="disclaimer">Research support only. Forecasts are probabilistic and '
    "do not constitute investment advice.</div>",
    unsafe_allow_html=True,
)
