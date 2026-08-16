# Gap Map: Ours × Karan × Design

**Purpose:** Freeze what we keep, rebuild, borrow, and defer so we finish the full production loop without drifting into endless training.

**Golden ticker for all demos until the loop works:** `NVDA`

**Spine:** Karan’s workflow (data → train → serve → analyze → monitor → docker).  
**Bar:** Our `DESIGN.md` / `implementation_plan.md` correctness and completeness.

---

## Legend

| Tag | Meaning |
|-----|---------|
| **KEEP** | Already in our repo; keep and use as-is (or only light polish) |
| **REBUILD** | Concept stays; our current implementation must be fixed or replaced |
| **BORROW** | Take from Karan’s work (shape/pattern), implement in *our* code |
| **DEFER** | Design mentions it; do it after the connected loop works |

---

## A. Stage map (what order we execute)

| Stage | Goal | User-visible proof |
|-------|------|--------------------|
| **S1** Forecast service | Honest predict path | `/predict` for NVDA with champion + last_close |
| **S2** Agent report loop | Analyze end-to-end | Streamlit or curl `/analyze` → report + chart |
| **S3** Eval & observability | See where AI/ML fails | Trace + agent eval + drift/metrics |
| **S4** Deploy | One-command platform | `docker compose up` full stack works |
| **S5** Cloud (optional) | Design Sprint 10 | K8s/AWS only after S4 is solid |

---

## B. Keep / Rebuild / Borrow / Defer

### 1. Market data & cleaning

| Item | Ours today | Karan | Decision | Notes |
|------|------------|-------|----------|-------|
| S&P 500 constituents + OHLCV ingestion | Stronger | Yahoo only, thin validation | **KEEP** | Use our `src/data/ingestion` |
| Schema / quality validation | Present | Minimal | **KEEP** | |
| Causal cleaning / ffill | Present | Weak | **KEEP** | |
| On-demand yfinance inside every predict | We mostly use stored features | Karan refetches always | **KEEP** (stored) | Borrow auto-refresh later if needed |

### 2. Features & Feast

| Item | Ours today | Karan | Decision | Notes |
|------|------------|-------|----------|-------|
| Feature engineering pipeline | Stronger (more features) | RSI/MACD only | **KEEP** | |
| Feast repo + apply/materialize | Present | Present but decorative | **REBUILD** | Make Feast *or* parquet the single authority for train/serve |
| Feast called but ignored at train/infer | Partial risk | Yes (ignore online features) | **REBUILD** | Do not copy Karan’s “log Feast, use Yahoo DF” |

### 3. Forecasting / training

| Item | Ours today | Karan | Decision | Notes |
|------|------------|-------|----------|-------|
| Parent → child transfer idea | Yes (two stacks) | Yes | **KEEP** idea | |
| Legacy `pipeline.py` / absolute-price path | Exists | Similar absolute multi-output | **REBUILD** / freeze | Do not use as default serve path |
| Full-universe returns + champion | Implemented, not fully promoted | No | **KEEP** (quality bar) | Promote artifacts before S2; don’t block S2 on all 503 |
| Chronological split + scale-invariant features | In full_universe | Random split + leakage | **KEEP** ours | Do not borrow Karan’s split |
| Persistence / parent / child gate | In full_universe | No gate | **KEEP** | |
| Auto-train on missing model | No | Yes | **BORROW** for S1/S2 | Fix task_id + Redis; no 4-worker races |
| Train inside API thread pool as “prod” | N/A | Yes | **DEFER** / avoid | V1: one worker or explicit job; durable queue later |

### 4. Model registry & inference API

| Item | Ours today | Karan | Decision | Notes |
|------|------------|-------|----------|-------|
| FastAPI `/predict`, `/health`, metrics | Present | Richer routes | **KEEP** + extend | |
| Local artifacts + promote script | Started | Local files only | **KEEP** / finish | `stock-promote` → `artifacts/models` |
| Champion manifest–driven load | Partially wired | Always child if exists | **KEEP** / finish | Require manifest; no auto-child |
| Strict predict DTO (champion, target_mode, version) | Partial | Loose | **REBUILD** | One contract for UI/agents |
| Redis prediction cache | Present (fail-open) | Broken import bug | **KEEP** ours pattern | Borrow key design; never copy stale import |
| `/train-parent`, `/train-child`, `/status` | Missing | Present | **BORROW** | Needed for Karan-like UX |
| MLflow tracking | Parent path yes; full_universe no | Tracking yes, registry unused | **KEEP** tracking | **DEFER** MLflow-as-serve-registry |

### 5. Agents & knowledge (Design Layer 2–3)

| Item | Ours today | Karan | Decision | Notes |
|------|------------|-------|----------|-------|
| LangGraph 4-node workflow | **Missing** | Present | **BORROW** | Performance → Market → Report → Critic |
| `/analyze` orchestration | **Missing** | Present | **BORROW** | Single user-facing entry |
| Finnhub + Yahoo news | **Missing** | Present | **BORROW** | |
| Structured agent outputs | N/A | Substring parse | **REBUILD** on borrow | Stance/confidence as fields; critic updates them |
| Feed analyst + news node outputs into report | N/A | Discarded | **REBUILD** on borrow | Don’t copy this bug |
| Qdrant report cache (ticker + TTL) | **Missing** | Present | **BORROW** | Call it report cache, not RAG |
| True RAG over filings/earnings | Design goal | Not real RAG | **DEFER** | After S2–S3 loop works |
| Agent run traces (node I/O) | **Missing** | Logs only | **BORROW** idea / **REBUILD** | Required for “why LLM wrong” |
| Agent eval heuristics | **Missing** | Keyword score | **BORROW** shell / **REBUILD** checks | Number match, stance consistency, sources |

### 6. UI

| Item | Ours today | Karan | Decision | Notes |
|------|------------|-------|----------|-------|
| Streamlit analyze UI | **Missing** | Present | **BORROW** | Poll **task_id**, timeout |
| Monitoring Streamlit | **Missing** | Present | **BORROW** for S3 | |
| History + forecast chart contract | Partial API | Broken history pass-through | **REBUILD** | History inside predictions DTO |

### 7. Monitoring & ops

| Item | Ours today | Karan | Decision | Notes |
|------|------------|-------|----------|-------|
| Prometheus metrics on API | Present | Present | **KEEP** | |
| Prediction logging JSONL | Present | Weak | **KEEP** | Add champion/model version |
| Drift | Evidently-style present | Custom, parent-only | **KEEP** ours + **BORROW** monitor UX | |
| `/ready` real checks | Partial | Always healthy | **REBUILD** | Models + features (+ Redis/Ollama when agents on) |
| Grafana provisioning | Started | Partial | **KEEP** / finish | |

### 8. Deploy & CI

| Item | Ours today | Karan | Decision | Notes |
|------|------------|-------|----------|-------|
| Dockerfile + compose (API/Redis/Prom/Grafana) | Present | Fuller stack | **KEEP** + **BORROW** Qdrant/UI | Wire Redis URL correctly |
| Frontend + monitoring containers | **Missing** | Present | **BORROW** in S4 | |
| GitHub Actions CI | Present | Absent in clone | **KEEP** | Add compose smoke later |
| K8s manifests | **Missing** | Present (Minikube-ish) | **DEFER** to S5 | After Compose works |
| Terraform / EKS | **Missing** | Docs only in clone | **DEFER** | Design Sprint 10 |

---

## C. Explicit “do not take” from Karan (hard rules)

1. Random time-series splits / scaler fit on all data / eval on train windows.  
2. Feast as theater (write/materialize but train/serve from Yahoo DF).  
3. `from module import redis_client` while client is still `None`.  
4. Discarding performance/market LLM outputs before the report.  
5. Critic rewrite without updating recommendation/confidence.  
6. Different `/analyze` shapes for cache hit vs miss.  
7. Dropping `task_id` so the UI polls forever.  
8. Claiming Evidently / Terraform / CI that aren’t in the running system.  
9. Health endpoint that always returns healthy.  
10. Four API workers racing local training jobs as “production.”

---

## D. Stage 1 Definition of Done (next build target)

S1 is done when **all** of these are true for `NVDA` (and optionally `AAPL`, `MSFT`):

1. Features exist on disk (and Feast sync works or is explicitly skipped with one documented fallback).  
2. At least one parent returns model is in `artifacts/models/parent_returns/` (promoted from Colab or a small local train).  
3. `champion_manifest.json` exists; loader never auto-picks a child without `champion == child`.  
4. `POST /predict` returns: predictions, `last_close`, `target_mode`, `champion`, model name/version.  
5. Persistence champion returns a flat forecast at `last_close`.  
6. Redis cache keys use ticker + effective horizon (+ model identity if available); fail-open if Redis down.  
7. `/ready` fails without usable weights + features.  
8. You can explain in one paragraph: which model served and why.

**Not required for S1:** all 503 tickers, perfect MAE, agents, Streamlit, Qdrant, cloud.

---

## E. What we do next (immediate sequence)

1. **Close S1 gaps only** — promote/serve path, contracts, readiness, Redis wiring.  
2. **Borrow S2 skeleton from Karan** — `/analyze`, graph, news, Qdrant cache, Streamlit — with our contract fixes.  
3. **S3** — traces, real agent checks, monitoring UI, prediction-vs-actual.  
4. **S4** — compose full stack like Karan’s topology.  
5. **S5** — K8s/cloud when local demo is solid.  
6. **Accuracy deep-dives** — only after S2, using logged forecasts vs actuals.

---

## F. Decision freeze

- **Workflow owner:** Karan’s end-to-end shape.  
- **ML correctness owner:** our full-universe / returns / champion approach (even if we start with a small ticker set).  
- **Product owner:** Design Layer 1 → 2 → 3, but RAG filings wait.  
- **No parallel AWS + full-universe + agents** until S1 DoD is checked off.

When S1 DoD is accepted, we start implementing S1 — still one stage at a time.
