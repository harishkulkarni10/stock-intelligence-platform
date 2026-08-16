# Stock Intelligence Platform

Production-style equity forecasting and agentic research: chronological LSTM
training, artifact serving, and a LangGraph analyze workflow that grounds LLM
reports in model forecasts and news tools.

## What's working

- **Data:** yfinance OHLCV + RSI/MACD → validated parquet feature store  
- **ML:** parent (`^GSPC`) / child transfer learning, price-space metrics, champion gates  
- **Serving:** FastAPI train/predict/status + Redis-backed task/prediction cache  
- **Agents:** `POST /analyze` — forecast tool → performance → news → report → critic  
- **UI:** minimal Streamlit analyze app (`frontend/app.py`)  
- **Colab:** GPU training notebook + `scripts/make_colab_bundle.py` handoff  

Still ahead: deeper monitoring/drift/eval, full compose polish, cloud/K8s.

## Layout

```text
backend/            FastAPI (health, ready, train, predict, analyze)
src/
  data/             ingestion + sequence preparation
  model/            LSTM, train, evaluate, save/load
  pipelines/        sip-data / sip-train / sip-predict
  agents/           tools, nodes, LangGraph analyze
  memory/           report cache (Redis TTL)
  monitoring/       drift / agent eval (next)
feature_store/      Feast definitions + offline parquet
frontend/           Streamlit analyze UI
notebooks/          Colab GPU training
outputs/            model artifacts (gitignored)
doc/                design notes
```

## Setup

```powershell
cd "Stock Intelligence Platform"
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev,agents,ui]"
copy .env.example .env
```

Install [Ollama](https://ollama.com) and pull a chat model (default in `.env`):

```powershell
ollama pull llama3.2:3b
```

Place trained weights under `outputs/parent/` (from Colab) or train locally.

## Quick start (API)

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict-parent `
  -ContentType application/json `
  -Body '{"ticker":"NVDA","horizon":5}'

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/analyze `
  -ContentType application/json `
  -Body '{"ticker":"NVDA"}'
```

Interactive docs: http://127.0.0.1:8000/docs  

Streamlit UI (API must be running):

```powershell
.\.venv\Scripts\streamlit.exe run frontend/app.py
```

## CLI pipelines

| Stage | Command |
|-------|---------|
| Build features | `sip-data build --tickers ^GSPC NVDA` |
| Inspect store | `sip-data inspect` |
| Materialize online | `sip-data materialize` |
| Train parent | `sip-train parent --source feature-store` |
| Train child | `sip-train child --ticker NVDA --source feature-store` |
| Predict | `sip-predict parent --ticker NVDA --horizon 5` |
| Serve | `sip-api` |

`--source feature-store` requires parquet. `auto` falls back to live download; `yfinance` forces live fetch.

## Colab training

1. `sip-data build` / `sip-data inspect` locally.  
2. Bundle with POSIX paths (**not** `Compress-Archive`):

```powershell
.\.venv\Scripts\python.exe scripts\make_colab_bundle.py
```

3. Upload `notebooks/colab_training.ipynb` + `stock-intelligence-colab.zip` to Colab (GPU).  
4. Download `sip_colab_outputs.zip` and unpack into `outputs/`.  

Serving uses `outputs/parent/model.pt`. A child folder is written only when the champion gate promotes the child.

## Analyze response (agents)

`POST /analyze` returns a single DTO:

- `predictions` — model forecast + history (from code, not the LLM)  
- `performance_analysis` — performance agent  
- `news_summary` — news agent  
- `final_report` / `recommendation` / `confidence` — report + critic  

Optional Redis report cache (`ANALYZE_CACHE_TTL_SECONDS`). News uses Finnhub when `FMI_API_KEY` is set; otherwise Yahoo.

## Compose

```powershell
docker compose up --build -d
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000/docs |
| Redis Insight | http://localhost:8001 |
| Qdrant | http://localhost:6333 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## Design principles

- Chronological train/validation/test splits (no random window leakage)  
- Scaler fit on train only; price-space evaluation + persistence baseline  
- Redis via `backend.state.get_redis()` (no import-time client binding)  
- `API_WORKERS=1` until an external job queue exists  
- Offline parquet is the training authority; online Feast/Redis is optional for serve  
- Agents interpret tool outputs; they do not invent forecast numbers  

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
