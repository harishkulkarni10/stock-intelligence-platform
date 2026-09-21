# Stock Intelligence Platform

An equity research desk: search a ticker, get a short-horizon price forecast, then specialist notes that interpret tool outputs and a research brief that stitches them. The app is research support only — it does not place trades or issue buy/sell recommendations.

**Live analyze path:** forecast (code) → Performance Analyst → Market Expert → Financial Analyst → Risk Analyst → Report.

The web UI also includes **Guide** (product help chat) and hover explanations on the summary chips (Trend, News, Confidence, Risk, Projected move). Those chip notes are written during the analyze run, not on hover.

---

## What you get

| Area | What it does |
| --- | --- |
| Forecast | Champion path from trained models (or a persistence fallback). Drawn on the chart with history. |
| Performance Analyst | Trend label and note grounded in the forecast path. Trend must match the numbers. |
| Market Expert | News tone and briefing from recent headlines. Returns `UNAVAILABLE` when nothing useful is found. |
| Financial Analyst | Fundamentals health, strengths, and weaknesses from a company snapshot. Does not invent statement figures. |
| Risk Analyst | Contained / Moderate / Elevated downside note from coded volatility, drawdown, news, financials, and forecast trust. |
| Report / Research brief | Stance-locked synthesis of the four specialist notes (executive summary, bull/bear, drivers, caveats). |
| Summary chips | Trend, News, Confidence, Risk, Projected move — with short explanations available on hover. |
| Guide | In-app assistant for how the desk works (agents, labels, UI). Out of scope for ticker picks and trading advice. |
| On your desk | In-session list of finished tickers for this browser visit (clears on refresh). |
| Analyze cache | Optional Redis cache for repeated tickers. UI badge + **Refresh analysis** to force a new run. |

---

## Prerequisites

- Python **3.11+**
- An LLM backend:
  - **Google AI Studio** key (`LLM_PROVIDER=google`), or
  - **[Ollama](https://ollama.com)** running locally (`LLM_PROVIDER=ollama`)
- Optional: **Redis** (analyze result cache)
- Optional: **Finnhub** API key for company news (`FMI_API_KEY` / `FINNHUB_API_KEY`)
- Trained model artifacts under `outputs/` (at least `outputs/parent/model.pt`). Without models, `/analyze` returns `missing_model`.

---

## Clone and run

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

Edit `.env`. Minimum useful settings:

```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemma-3-12b-it

# Or local Ollama:
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.2:3b
```

If using Ollama, pull a chat model:

```bash
ollama pull llama3.2:3b
```

### 3. Start the app

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**macOS / Linux**

```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000**

| Link | Purpose |
| --- | --- |
| http://127.0.0.1:8000 | Research desk UI |
| http://127.0.0.1:8000/api | API route map |
| http://127.0.0.1:8000/docs | OpenAPI docs |
| `GET /ready` | Readiness check |

A full analyze run usually takes under a minute with a cloud LLM (forecast + three agents + chip explanations). Local Ollama can be slower.

---

## Using the UI

1. Enter a ticker and run analysis.
2. Read the summary chips, company overview, and forecast chart.
3. Open the agent tabs for Performance, Market, and Financial notes.
4. Hover (or tap) a summary chip for a short explanation of that label.
5. Use **Guide** for product questions (how agents work, how to read a run).
6. Use **On your desk** to reopen tickers finished in this session.
7. Use **Refresh analysis** when you want a brand-new run instead of a recent saved result.

---

## Project layout

```text
backend/              FastAPI app, schemas, HTTP middleware
frontend/web/         Main research desk UI (HTML / CSS / JS)
frontend/app.py       Optional Streamlit UI
src/
  data/               Ingestion and sequence prep
  model/              LSTM train / evaluate / save-load
  pipelines/          sip-data / sip-train / sip-predict
  market/             Company profile + fundamentals snapshot
  agents/             Tools, nodes, guardrails, Guide, LangGraph analyze
  memory/             Analyze result cache (Redis TTL)
  monitoring/         Agent fixture evals
logger/               Structured logging + request context
feature_store/        Feast definitions + offline parquet
notebooks/            Colab GPU training
scripts/              CLI runners
tests/                Pytest suite
outputs/              Model artifacts (gitignored)
```

---

## CLI (no UI)

From the project root with the venv active:

```powershell
# Guardrail fixtures (no live LLM / news required for eval-only)
python scripts\run_analyze_agents.py --eval-only

# Forecast + agents 1–3
python scripts\run_analyze_agents.py --ticker NVDA

# Full analyze path (same idea as POST /analyze)
python scripts\run_analyze_agents.py --full-analyze --ticker NVDA
```

On macOS/Linux use `python scripts/run_analyze_agents.py ...`.

---

## Data / train / predict

| Stage | Command |
| --- | --- |
| Build features | `sip-data build --tickers ^GSPC NVDA AAPL MSFT` |
| Inspect store | `sip-data inspect` |
| Train parent | `sip-train parent --source feature-store` |
| Train child | `sip-train child --ticker NVDA --source feature-store` |
| Predict | `sip-predict best --ticker NVDA --horizon 5` |

---

## API overview

### `POST /analyze`

Runs forecast + Performance + Market + Financial + Risk + Report.

Notable response fields:

- `predictions` — champion forecast and history (code, not LLM)
- `performance_analysis` / `performance_trend` / `performance_guardrail_ok`
- `news_summary` / `news_sentiment` / `news_guardrail_ok`
- `financial_analysis` / `financial_health` / `financial_guardrail_ok`
- `financials` — normalized fundamentals snapshot (code, not LLM)
- `company` — company profile for the results header
- `metric_explanations` — short blurbs for Trend, News, Confidence, Risk, Projected move
- `final_report` / `draft_report` — synthesis research brief
- `recommendation` — `BULLISH` / `BEARISH` / `NEUTRAL` (SIDEWAYS → NEUTRAL; locked from Performance trend)
- `confidence` — `Low` when serving persistence or repaired performance output; otherwise `Medium`
- `report_guardrail_ok` / `report_repaired`
- `risk_analysis` / `risk_level` / `risk_guardrail_ok`
- `risk` — coded risk metrics snapshot
- `mode` — `performance_news_financial_risk_report`
- `cached` / `cache_age_seconds` / `cache_ttl_seconds`

Request flag: `force_refresh: true` bypasses the analyze cache.

Cache prefix: `analyze-perf-news-fin-risk-report-v1` (TTL from `ANALYZE_CACHE_TTL_SECONDS`, default 1 hour).

### `POST /help-chat`

Guide product Q&A. Body: `{ "message": "...", "history": [] }`.  
Replies are grounded in a fixed product knowledge pack. Off-topic / trading questions return out of scope.

### News and fundamentals sources

- **News:** Finnhub company-news when keyed; otherwise Yahoo with ticker relevance filtering.
- **Fundamentals:** Yahoo / yfinance snapshot. Thin coverage becomes `partial` or `UNAVAILABLE`.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```bash
pytest -q
```

---

## Design notes

- Forecast prices come from code (trained path or persistence). Agents interpret tool outputs; they do not invent forecast prices, headlines, or fundamentals figures.
- Performance: trend label must match the forecast path.
- Market: drivers must be grounded in fetched headlines.
- Financial: health label and magnitudes must match the fundamentals facts object.
- Risk: Contained / Moderate / Elevated must match coded severity; no invented vol/drawdown; no trade advice.
- Chip explanations are produced once per analyze run by the Performance Analyst module and stored on the response.
- Chronological train / validation / test splits; scaler fit on train only; price-space metrics with a persistence baseline.
- Structured JSON logs with request context for local and production runs.

---

## License

MIT (see `pyproject.toml`).
