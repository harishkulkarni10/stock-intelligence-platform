# Stock Intelligence Platform

Equity forecasting with guardrailed agents on champion model paths.
Forecasts are produced by code (child LSTM / parent / persistence). LLMs only interpret tool outputs.

**Live path:** forecast → Performance Analyst (agent 1) → Market Expert / news (agent 2) → Financial Analyst (agent 3).

## Clone and run

### Prerequisites

- Python 3.11+ recommended
- [Ollama](https://ollama.com) installed and running **or** a Google AI Studio key for Gemma (`LLM_PROVIDER=google`)
- Optional: Redis (analyze result cache)
- Optional: Finnhub API key for company news (`FMI_API_KEY` / `FINNHUB_API_KEY` in `.env`)



### 1. Clone

```bash
git clone https://github.com/harishkulkarni10/stock-intelligence-platform.git
cd stock-intelligence-platform
```



### 2. Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,agents,ui]"
copy .env.example .env
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,agents,ui]"
cp .env.example .env
```

Pull a local chat model (only if using Ollama):

```bash
ollama pull llama3.2:3b
```

**LLM switch** (in `.env`):

```env
# Use Google Gemma (faster) or local Ollama
LLM_PROVIDER=google
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemma-3-12b-it

# Or flip back to local:
# LLM_PROVIDER=ollama
```

Place trained artifacts under `outputs/parent/` (at least `model.pt`) so forecasts can run. Child tickers may live under `outputs/<TICKER>/`. Without models, `/analyze` returns `missing_model`.

### 3. Start the app

Keep Ollama running, then:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**macOS / Linux**

```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

- UI: search a ticker → **Run agents** (forecast + agent 1 + agent 2)
- API map: [http://127.0.0.1:8000/api](http://127.0.0.1:8000/api)
- OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Agent runs often take about a minute (two local LLM calls).

## Layout

```text
backend/            FastAPI (REST API + web UI)
src/
  data/             ingestion + sequence preparation
  model/            LSTM, train, evaluate, save/load
  pipelines/        sip-data / sip-train / sip-predict
  agents/           tools, nodes, guardrails, LangGraph analyze
  memory/           report cache (Redis TTL)
  monitoring/       agent fixture evals
feature_store/      Feast definitions + offline parquet
frontend/web/       Web UI (light/dark) for agents 1–3
frontend/app.py     Optional Streamlit UI
notebooks/          Colab GPU training
scripts/            CLI runners
outputs/            model artifacts (gitignored)
```



## CLI (no UI)

From the project root with the venv active:

```powershell
# Guardrail fixtures (no Ollama / no live news)
python scripts\run_analyze_agents.py --eval-only

# Forecast + agents 1–3
python scripts\run_analyze_agents.py --ticker NVDA

# Full analyze path (same as POST /analyze)
python scripts\run_analyze_agents.py --full-analyze --ticker NVDA
```

On macOS/Linux use `python scripts/run_analyze_agents.py ...`.

## Data / train / predict pipelines


| Stage          | Command                                                |
| -------------- | ------------------------------------------------------ |
| Build features | `sip-data build --tickers ^GSPC NVDA AAPL MSFT`        |
| Inspect store  | `sip-data inspect`                                     |
| Train parent   | `sip-train parent --source feature-store`              |
| Train child    | `sip-train child --ticker NVDA --source feature-store` |
| Predict        | `sip-predict best --ticker NVDA --horizon 5`           |




## Analyze response

`POST /analyze` returns:

- `predictions` — champion forecast + history (code, not LLM)
- `performance_analysis` / `performance_trend` / `performance_guardrail_ok`
- `news_summary` / `news_sentiment` / `news_guardrail_ok`
- `financial_analysis` / `financial_health` / `financial_guardrail_ok`
- `financials` — normalized Yahoo fundamentals snapshot (code, not LLM)
- `recommendation` — BULLISH / BEARISH / NEUTRAL (SIDEWAYS → NEUTRAL)
- `confidence` — Low when serving persistence or repaired output
- `mode` — `performance_news_financial`
- `cached` / `cache_age_seconds` / `cache_ttl_seconds` — Redis analyze cache metadata

Redis cache prefix: `analyze-perf-news-fin-v1` (TTL from `ANALYZE_CACHE_TTL_SECONDS`, default 1h).  
UI shows a cache badge; **Refresh analysis** sends `force_refresh: true` to bypass Redis and re-run the agents.

News: Finnhub company-news when keyed; otherwise Yahoo with ticker relevance filtering. If nothing relevant remains, agent 2 returns `UNAVAILABLE`.

Fundamentals: Yahoo / yfinance snapshot. Thin coverage becomes `partial` or `UNAVAILABLE` — agent 3 must not invent statement figures.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```bash
pytest -q
```



## Design notes

- Chronological train/validation/test splits
- Scaler fit on train only; price-space metrics + persistence baseline
- Agents interpret tools; they do not invent forecast prices, headlines, or fundamentals
- Agent 1: trend must match numbers
- Agent 2: drivers must be grounded in fetched news
- Agent 3: health label and magnitudes must match the fundamentals facts object

