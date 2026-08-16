# Greenfield scaffold — Karan-shaped, ops-correct

Reference: `Stock Agent Ops clone/stock-agent-ops`  
Design notes: copy `DESIGN.md`, `implementation_plan.md`, `GAP_MAP_KARAN_VS_OURS.md` into new repo `doc/` when created.

## Target tree (match Karan first)

```text
stock-intelligence-platform/
  backend/           main.py, api.py, tasks.py, state.py, schemas.py, rate_limiter.py, Dockerfile
  src/
    config.py
    data/            ingestion.py, preparation.py
    model/           definition.py, training.py, evaluation.py, saving.py
    pipelines/       training_pipeline.py, inference_pipeline.py
    inference.py
    agents/          graph.py, nodes.py, tools.py
    memory/          semantic_cache.py
    monitoring/      drift.py, agent_eval.py
    utils.py
  feature_store/     feature_store.yaml, features.py
  frontend/          app.py, Dockerfile
  monitoring_app/    app.py, Dockerfile
  logger/            logger.py
  prometheus/        prometheus.yml
  grafana/provisioning/datasources/
  k8s/
  doc/               commands.md, system_design.md (lean)
  outputs/           gitignored
  logs/              gitignored
  docker-compose.yml
  run_docker.sh
  pyproject.toml
  .env.example
  .gitignore
  README.md
```

## Ops fixes to bake in from day one (from Karan audit)

| Issue in reference | Our rule |
|--------------------|----------|
| `from backend.state import redis_client` at import | Use `backend.state.redis_client` or getter inside functions |
| `REDIS_HOST` in compose but hardcoded `redis` in startup | Read env in `backend/main.py` |
| `/health` always 200 | `/ready` checks redis, features, parent weights, ollama (when agents on) |
| `DELETE /system/reset` open | Dev-only flag or auth; not in prod compose |
| Grafana `admin/admin` | Env vars; provision datasource in `grafana/provisioning/` |
| No `grafana/provisioning` in repo | Add datasource yaml before claiming Grafana works |
| 4 uvicorn workers + in-process training | Compose: 1 worker until job queue exists |
| Monitoring app without outputs mount on k8s | Same PVC on fastapi + monitoring_app |
| Ollama only on host | Document host URL in compose; k8s needs explicit env/service |
| Evidently claimed, custom drift only | Name drift honestly; schedule later |
| No tests / CI | Add `tests/` + `.github/workflows/ci.yml` in sprint 1 |
| `pyproject.toml` vs `backend/requirements.txt` drift | One lockfile source (uv); Docker uses same |
| Case mismatch `outputs/AAPL` vs `outputs/aapl` | Normalize ticker paths once |
| Agent history dropped before UI | Single analyze DTO; history inside `predictions` |
| Task ID dropped on 202 | Pass `task_id` through analyze + Streamlit poll |

## Build order

1. Skeleton + git init + `.gitignore`  
2. `backend/main.py` health + empty router  
3. `src/config.py` + `logger/`  
4. Data ingestion (yfinance + RSI/MACD) — our validation, Karan file names  
5. Feast write path (batch job, not every predict)  
6. Train parent/child pipelines — chronological split, no random_split  
7. Predict + Redis + `/status`  
8. Agents + `/analyze`  
9. Streamlit + monitoring_app  
10. Compose full stack  
11. Tests + CI  
12. K8s stubs last  

## First session deliverable

Only folders, empty `__init__.py` where needed, stub `backend/main.py` (`GET /health`), `.gitignore`, `.env.example`, README with compose ports table.

No training code until tree is approved.
