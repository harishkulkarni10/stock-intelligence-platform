# Stock Intelligence Platform

Production-style equity forecasting with a guardrailed Performance Analyst (agent 1).
Champion forecast (child LSTM / parent / persistence) is produced by code; the LLM
only interprets those numbers.

## What's working now (agent-1 production path)

- **Data:** yfinance OHLCV + RSI/MACD → validated parquet feature store
- **ML:** parent (`^GSPC`) / child transfer learning, price-space metrics, champion gates
- **Serving:** FastAPI train/predict/status + Redis task/prediction cache
- **Agent 1:** `POST /analyze` → `get_forecast` → Performance Analyst harness (trend / range / caution + guardrails + fixture evals)
- **UI:** Streamlit shows forecast chart + agent 1 only (`frontend/app.py`)
- **Colab:** GPU training notebook + `scripts/make_colab_bundle.py`

Agents 2–4 (news / report / critic) remain in code via `build_full_graph()` but are **not** in the UI/API default path yet.

Still ahead: agents 2–4 productization, latency/throughput SLOs, deeper drift dashboards, cloud/K8s.

## Layout

```text
backend/            FastAPI (health, ready, train, predict, analyze)
src/
  data/             ingestion + sequence preparation
  model/            LSTM, train, evaluate, save/load
  pipelines/        sip-data / sip-train / sip-predict
  agents/           tools, nodes, guardrails, LangGraph analyze
  memory/           report cache (Redis TTL)
  monitoring/       performance agent fixture evals
feature_store/      Feast definitions + offline parquet
frontend/           Streamlit agent-1 UI
notebooks/          Colab GPU training
scripts/            colab bundle, run_performance_agent
outputs/            model artifacts (gitignored)
doc/                design notes
```

## Setup

```powershell
cd "D:\Harish\AI projects\Stock-Agent-Ops-Cursor\Stock Intelligence Platform"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,agents,ui]"
copy .env.example .env
```

Install [Ollama](https://ollama.com) and pull a chat model:

```powershell
ollama pull llama3.2:3b
```

Place trained weights under `outputs/parent/` (from Colab) or train locally.

## Quick start (API + UI)

Terminal 1 — API:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

Open http://localhost:8501 → enter ticker → **Run agent 1**.

CLI checks (no UI):

```powershell
python scripts\run_performance_agent.py --eval-only
python scripts\run_performance_agent.py --ticker NVDA
```

## Analyze response (agent 1)

`POST /analyze` returns:

- `predictions` — champion forecast + history (**code**, not LLM)
- `performance_analysis` / `performance_trend` / `performance_guardrail_ok`
- `recommendation` — BULLISH / BEARISH / NEUTRAL (SIDEWAYS → NEUTRAL)
- `confidence` — Low when serving persistence or repaired output
- `mode` — `performance_only`

Redis cache key prefix: `analyze-perf-v1`.

## CLI pipelines

| Stage | Command |
|-------|---------|
| Build features | `sip-data build --tickers ^GSPC NVDA` |
| Inspect store | `sip-data inspect` |
| Train parent | `sip-train parent --source feature-store` |
| Train child | `sip-train child --ticker NVDA --source feature-store` |
| Predict | `sip-predict best --ticker NVDA --horizon 5` |

## Design principles

- Chronological train/validation/test splits (no random window leakage)
- Scaler fit on train only; price-space evaluation + persistence baseline
- Agents interpret tool outputs; they do not invent forecast numbers
- Agent 1 hard guardrails: trend must match numbers; no invented prices
- Offline parquet is the training authority

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
