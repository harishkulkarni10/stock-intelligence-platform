# Stock Intelligence Platform

Forecasting + agentic equity research. Layout mirrors [kmeanskaran/stock-agent-ops](https://github.com/kmeanskaran/stock-agent-ops); correctness and ops rules are ours.

## Status

Stage 1 forecasting is complete: chronological ingestion, parent/child LSTM training,
price-space evaluation, champion gates, artifacts, inference, and API orchestration.
Agent workflows remain Stage 2.

## Layout

```text
backend/           FastAPI
src/               data, model, pipelines, agents, memory, monitoring
feature_store/     Feast
frontend/          Streamlit (Stage 2)
monitoring_app/    Streamlit (Stage 3)
prometheus/ grafana/ k8s/
doc/               design + reference notes
outputs/ logs/     runtime (gitignored)
```

## Setup

```bash
cd "Stock Intelligence Platform"
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

## Run API (local)

```bash
uvicorn backend.main:app --reload --port 8000
```

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/
```

## Pipelines

Every stage is a CLI entrypoint; no ad-hoc scripting required.

| Stage | Command |
|-------|---------|
| Build feature store | `sip-data build --tickers ^GSPC NVDA` |
| Inspect coverage and splits | `sip-data inspect` |
| Materialize to online store | `sip-data materialize` |
| Train parent | `sip-train parent --source feature-store` |
| Train child | `sip-train child --ticker NVDA --source feature-store` |
| Predict | `sip-predict child --ticker NVDA --horizon 5` |
| Analyze (agents) | `POST /analyze` with `{"ticker":"NVDA"}` |
| Serve | `sip-api` |

`--source feature-store` fails loudly when data is absent. `auto` falls back to live
ingestion, and `yfinance` bypasses the store entirely.

## Train on Google Colab

1. Ensure the offline store exists (`sip-data build` / `sip-data inspect`).
2. Build a Colab-safe ZIP with POSIX paths (**do not** use PowerShell `Compress-Archive`):

```powershell
.\.venv\Scripts\python.exe scripts\make_colab_bundle.py
```

3. Upload `notebooks/colab_training.ipynb` to Colab, select a GPU runtime, then upload
   `stock-intelligence-colab.zip` when the notebook asks.
4. Train with `source="feature-store"` (uses the parquet in the ZIP; no live rebuild).
5. Download `sip_colab_outputs.zip` and import into the project:

```powershell
Expand-Archive -Path sip_colab_outputs.zip -DestinationPath outputs -Force
```

Serving loads `outputs/parent/model.pt` and, only if promoted,
`outputs/<TICKER>/model.pt`. Local retraining is not required.

## Compose (infra)

```bash
docker compose up --build -d
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000/docs |
| Redis Insight | http://localhost:8001 |
| Qdrant | http://localhost:6333 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Frontend / monitoring containers land in Stage 2–3.

## Reference

- Local clone: `../Stock Agent Ops clone/stock-agent-ops` (or your path)
- `doc/KARANS_WORK_EXPLAINED.md`
- `doc/GAP_MAP_KARAN_VS_OURS.md`
- `doc/GREENFIELD_SCAFFOLD.md`
- `doc/DESIGN.md`

## Build order

1. Scaffold ✓
2. Data ingestion + Feast ✓
3. Parent/child train + predict + Redis ✓ ← current
4. Agents + `/analyze` + Streamlit
5. Monitoring + drift/eval  
6. Full compose + CI  
7. K8s / cloud  

## Ops rules (non-negotiable)

- Redis via `backend.state.get_redis()`, not import-time binding  
- `API_WORKERS=1` until a job queue exists  
- Chronological ML splits; no random window split  
- Feast or parquet is real train/serve source — not decoration  
- Single analyze/predict DTO; pass `task_id` on training responses  
