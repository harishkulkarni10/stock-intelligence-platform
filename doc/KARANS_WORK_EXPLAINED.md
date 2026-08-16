# Karan's Stock-Agent-Ops — Complete Code-Grounded Walkthrough

## Why this document exists

The repository inside `stock-agent-ops/` is Karan's work. We will use it as a
reference implementation for our Stock Intelligence Platform, not as code to
copy without understanding.

This guide answers four questions:

1. What does each part of Karan's repository do?
2. How does a request move through data, ML, agents, storage, UI, and monitoring?
3. Which README/design claims are genuinely implemented?
4. Which ideas should we adopt, improve, or avoid in our own project?

Paths in this document are relative to:

```text
Stock Agent Ops clone/stock-agent-ops/
```

---

## 1. The honest one-paragraph assessment

Karan's work is a useful **end-to-end learning reference** because it connects
Yahoo Finance, feature engineering, Feast, an LSTM parent/child training flow,
MLflow, FastAPI, Redis, LangGraph, Ollama, Qdrant, Streamlit, Prometheus,
Grafana, Docker Compose, and Kubernetes.

However, it is not a fully validated production platform. It has no automated
test suite in this clone, no checked-in CI/CD workflow or Terraform files,
time-series leakage in training/evaluation, weak model and agent evaluations,
and several places where Feast, MLflow registry, semantic caching, and
Kubernetes are present more as demonstrations than robust production
integrations.

The right lesson is therefore:

> Learn the system boundaries and orchestration from Karan's work, but verify
> the statistical correctness, reliability, security, and deployment claims
> before adopting the implementation.

---

## 2. Repository map

### Application entry points

- `backend/main.py` — creates the FastAPI application, installs routes and
  Prometheus instrumentation, initializes directories, and connects to Redis.
- `backend/api.py` — all API endpoints and most service orchestration.
- `frontend/app.py` — user-facing Streamlit stock analysis interface.
- `monitoring_app/app.py` — Streamlit diagnostics interface.
- `main.py` — older/simple local entry point; it is not the Docker Compose API
  entry point.

### Data and features

- `src/data/ingestion.py` — downloads OHLCV data, computes RSI/MACD, writes the
  shared Feast parquet file, and invokes Feast CLI commands.
- `src/data/preparation.py` — converts a dataframe into 60-day input windows and
  5-day targets.
- `feature_store/features.py` — Feast entity, file source, and feature view.
- `feature_store/feature_store.yaml` — local registry, file offline store, and
  Redis online store.

### ML

- `src/config.py` — model/data defaults.
- `src/model/definition.py` — multi-output LSTM architecture.
- `src/model/training.py` — optimizer, loss, validation, scheduler, early stop.
- `src/model/evaluation.py` — MSE/RMSE/R² and plots.
- `src/pipelines/training_pipeline.py` — parent and child training workflows.
- `src/inference.py` — turns one model output into five future business-day
  predictions.
- `src/pipelines/inference_pipeline.py` — loads local artifacts, fetches current
  data, optionally reads Feast online features, and calls inference.
- `compare_transfer_learning.py` — experiment-oriented transfer-learning
  comparison script.

### Agents and memory

- `src/agents/graph.py` — LangGraph state, graph topology, semantic-cache lookup,
  prediction retrieval, graph invocation, and cache save.
- `src/agents/nodes.py` — performance analyst, market expert, report generator,
  and critic.
- `src/agents/tools.py` — calls the prediction API and fetches Finnhub/Yahoo news.
- `src/memory/semantic_cache.py` — Qdrant collection, ticker/TTL filtering,
  report save, and retrieval.

### Operations and observability

- `backend/tasks.py` — thread-pool execution, background training, Redis task
  status, prediction cache, and system metrics.
- `backend/state.py` — shared Redis client, executor, and Prometheus metrics.
- `backend/rate_limiter.py` — Redis fixed-window rate limiter.
- `src/monitoring/drift.py` — custom recent-vs-reference distribution checks.
- `src/monitoring/agent_eval.py` — heuristic report evaluation.
- `prometheus/prometheus.yml` — Prometheus scrape configuration.
- `grafana/` — Grafana state/provisioning.
- `docker-compose.yml` — seven-service local stack.
- `k8s/*.yaml` — Minikube-style Kubernetes manifests.
- `run_docker.sh`, `run_k8s.sh` — local orchestration scripts.
- `doc/AWS.md` — proposed AWS/Terraform/CI workflow, not an implementation in
  this clone.

---

## 3. High-level architecture

```text
User
  |
  v
Streamlit frontend (:8501)
  |
  | POST /analyze
  v
FastAPI (:8000)
  |
  +--> Redis
  |      - prediction cache
  |      - training task status
  |      - rate limits
  |
  +--> Local model artifacts in outputs/
  |      - parent LSTM
  |      - per-ticker child LSTM
  |      - scalers and evaluation files
  |
  +--> Yahoo Finance
  |      - training and current OHLCV
  |
  +--> Feast
  |      - shared parquet offline data
  |      - Redis online features
  |
  +--> LangGraph
         |
         +--> prediction API call
         +--> Finnhub/Yahoo news
         +--> Ollama LLM
         +--> Qdrant semantic report cache

Prometheus scrapes FastAPI metrics
Grafana visualizes Prometheus
Monitoring Streamlit reads API results and shared outputs/
```

Although the documentation calls this microservices architecture, most business
logic lives inside one FastAPI container. Redis, Qdrant, Prometheus, Grafana,
and the two Streamlit apps are separate services. Training, inference, agents,
and monitoring logic are Python modules in the API process, not independent
services.

---

## 4. The complete `/analyze` request journey

This is the most important workflow because it connects almost every system.

### Step 1: User submits a ticker

`frontend/app.py` sends:

```json
{
  "ticker": "NVDA",
  "thread_id": "<streamlit-session-uuid>"
}
```

to `POST /analyze`.

### Step 2: FastAPI delegates to the agent system

`backend/api.py::analyze()` validates the Pydantic request and calls:

```text
src.agents.graph.analyze_stock(ticker, thread_id)
```

The API endpoint itself is synchronous. LangGraph and the LLM run before the
HTTP response completes.

### Step 3: Qdrant semantic cache lookup

`analyze_stock()`:

1. Creates an Ollama embedding for `"Analysis report for NVDA"`.
2. Opens the Qdrant `dataset_cache` collection.
3. Filters by exact ticker and entries newer than 24 hours.
4. Accepts results with cosine score greater than `0.95`.
5. Returns the newest matching cached report when found.

Important nuance: for one exact ticker the query text is always effectively the
same. This behaves more like a ticker-specific report cache with embeddings than
a genuinely varied semantic query system. A normal keyed cache could satisfy
this use case more simply unless future user questions differ semantically.

### Step 4: Fetch prediction through FastAPI

On a cache miss, `src/agents/tools.py::fetch_prediction_data()` makes an HTTP
request back to the same API:

```text
POST /predict-child {"ticker": "NVDA"}
```

This loopback call creates a notable deployment concern:

- In Docker, `API_BASE_URL` defaults to `http://localhost:8000`, which refers to
  the same API container and can work.
- With four Uvicorn workers, blocking internal HTTP requests consume another
  worker.
- In constrained deployments this pattern can cause avoidable latency or
  deadlock risk.
- A direct internal service call would be simpler unless the prediction
  component is separated into its own service.

### Step 5: Prediction cache or model inference

`backend/api.py::predict_child_endpoint()` checks Redis key:

```text
predict_child_nvda
```

The cache TTL is 86,400 seconds. On a miss:

1. Load `outputs/NVDA/NVDA_child_model.pt`.
2. Load `outputs/NVDA/NVDA_child_scaler.pkl`.
3. Download current NVDA history from Yahoo Finance.
4. Recompute RSI and MACD.
5. Save/materialize features through Feast as a side effect.
6. Scale the latest 60 rows.
7. Produce five days × seven features.
8. Inverse-transform predictions.
9. Return future OHLCV and recent history.
10. Cache the result in Redis.

### Step 6: Auto-training when model is missing

If prediction fails because the child model is absent:

1. Check whether the parent `^GSPC` model exists.
2. If parent is missing, schedule parent training.
3. Otherwise schedule child training in a thread pool.
4. Return HTTP `202` and a training task identifier.
5. After child training, run a chained prediction and populate Redis.
6. The frontend polls `/status/{task_id}` every three seconds.
7. When complete, Streamlit reruns the analysis.

This is the project's “auto-healing” mechanism. It is convenient for a demo,
but production training should usually run in a durable job system rather than
inside an API process. Process restarts lose the in-memory asyncio task, and
four concurrent thread workers are not a scalable training queue.

### Step 7: LangGraph agent sequence

When predictions are available, the graph executes:

```text
START
  -> performance analyst
  -> market expert
  -> report generator
  -> critic
  -> END
```

This is a fixed linear workflow. It does not dynamically choose agents, retry
nodes, route based on critic quality, or loop until an acceptance threshold.

### Step 8: Cache and return report

The final graph output, recommendation, confidence, last price, and predictions
are stored in Qdrant with the query embedding and timestamp. FastAPI returns the
result to Streamlit, which renders report text, recommendation, confidence, and
a forecast chart.

---

## 5. Data ingestion and Feast

### What `fetch_ohlcv()` really does

`src/data/ingestion.py::fetch_ohlcv()`:

1. Downloads daily adjusted Yahoo Finance data.
2. Flattens yfinance MultiIndex columns.
3. Retains date, OHLCV.
4. Computes 14-day RSI.
5. Computes MACD as EMA(12) minus EMA(26).
6. Drops missing values.
7. Performs basic numeric and minimum-row checks.
8. Appends data to `feature_store/data/features.parquet`.
9. Deduplicates by ticker and timestamp.
10. Runs `feast apply`.
11. Runs `feast materialize-incremental`.

### Feast definitions

`feature_store/features.py` defines:

- Entity: `ticker`.
- Source: `/app/feature_store/data/features.parquet`.
- Features: Open, High, Low, Close, Volume, RSI14, MACD.
- Feature view: `stock_stats`.
- Online serving enabled.

`feature_store/feature_store.yaml` uses:

- SQLite registry.
- File offline store.
- Redis online store.

### What Feast is and is not doing here

Feast stores and serves the features, but the actual parent/child training code
does **not** call Feast historical retrieval. It trains directly from the
dataframe returned by Yahoo Finance.

Inference calls Feast online retrieval, logs the returned values, then ignores
them and predicts from the Yahoo dataframe.

Therefore:

> Feast is integrated operationally, but it is not the authoritative
> train/serve feature path.

This weakens the README claim of training-serving consistency.

### Operational concerns

- `fcntl` file locking is Unix-specific and fails on native Windows.
- Every data fetch runs Feast apply/materialization, including inference. This
  is expensive and mixes deployment/data-pipeline responsibilities.
- The parquet file is shared by all tickers and rewritten after concatenation.
- Multiple API workers can race around Feast registry/materialization.
- The Feast source path is hardcoded to `/app/...`, making non-container local
  execution inconsistent.
- A 100-year feature TTL is effectively “never expire.”

### Better separation

Use three explicit jobs:

1. Ingestion/feature pipeline writes validated, versioned features.
2. Feast apply/materialization runs during deployment or scheduled updates.
3. Training and inference read from Feast APIs without rewriting the feature
   repository.

---

## 6. LSTM model: what it predicts

`src/model/definition.py::LSTMModel` receives:

```text
(batch, 60 days, 7 features)
```

The three-layer LSTM emits its final hidden representation. A linear layer
produces:

```text
5 future days × 7 features
```

The model therefore jointly forecasts:

- Open
- High
- Low
- Close
- Volume
- RSI14
- MACD

This is a direct multi-horizon model, not an autoregressive one-step model.

### Why the name is slightly misleading

`src/inference.py::predict_one_step_and_week()` sounds iterative, but it performs
one forward pass that predicts all five future rows.

### Training objective

All seven standardized outputs are optimized with one MSE loss. This means:

- The model spends capacity forecasting indicators that are derived from price.
- All standardized dimensions receive similar loss weight.
- The business target—future close—is not isolated.
- No directional or return-specific objective is used.

---

## 7. Parent training

`src/pipelines/training_pipeline.py::train_parent()`:

1. Uses `^GSPC`, the S&P 500 index—not all S&P 500 constituents.
2. Fetches data from August 19, 2004 onward.
3. Fits one StandardScaler on the **entire dataframe**.
4. Creates all 60-to-5 windows.
5. Randomly splits windows 80/20.
6. Trains for up to 20 epochs.
7. Saves model and scaler under `outputs/parent/`.
8. Evaluates on all windows from the same dataframe.
9. Logs parameters, metrics, files, and plots to MLflow.

### Statistical correctness problems

#### Leakage from scaler fitting

The scaler sees future validation/evaluation rows before training.

#### Random time-series split

`random_split()` mixes earlier and later overlapping windows. Adjacent windows
share most of their 60-day input and five-day target periods, so train and
validation can be nearly duplicates.

#### Evaluation uses the training history

`evaluate_model_temp()` rebuilds and scores every possible window, including
training windows. This is not a held-out test.

#### Best model is not restored

Early stopping tracks the best validation loss but never snapshots/restores the
best weights. The returned model is the final epoch before stopping.

#### Metrics are in standardized mixed-feature space

MSE, RMSE, and R² are computed jointly across scaled OHLCV values, not actual
close-price units. Volume and price semantics are mixed.

For an honest forecasting experiment, use chronological train/validation/test
boundaries, fit preprocessing only on training, prevent overlapping windows
across boundaries, restore the best checkpoint, and report price/return metrics
against baselines.

---

## 8. Child transfer learning

`train_child(ticker)`:

1. Downloads the child ticker.
2. Fits a new child-specific scaler.
3. Builds and randomly splits child sequences.
4. Loads the parent LSTM weights.
5. Uses one configured strategy:
   - `freeze`: freeze all parameters whose name contains `lstm`.
   - `fine_tune`: train all parameters with a lower learning rate.
6. Saves a child model and scaler.
7. Evaluates and logs it to MLflow.

### What is transferred

The network structure and learned weights transfer. The child's own scaler is
used for both child training and serving.

Because each ticker is standardized separately, the parent learns patterns in
normalized feature space. This makes transfer more plausible than feeding raw
price levels, although the data leakage concerns remain.

### What is missing

- No comparison against training the child from scratch in the serving gate.
- No persistence/naive baseline.
- No rule that rejects a child when it is worse than parent or baseline.
- No champion manifest.
- No full-universe parent; the parent sees only the market index.
- No automated scheduled retraining or champion promotion.

`compare_transfer_learning.py` appears intended for strategy comparison, but the
main API always serves the saved child artifact without a quality gate.

---

## 9. MLflow and DagsHub

`src/utils.py::setup_dagshub_mlflow()` configures tracking. Training runs log:

- Hyperparameters.
- Per-epoch train/validation loss and learning rate.
- Aggregate MSE/RMSE/R².
- `.pt` model state.
- scaler.
- metrics JSON.
- prediction and residual plots.

The local filesystem remains the actual serving registry.

Although `_safe_promote_to_production()` exists, the calls that register models
are commented out. `src/pipelines/inference_pipeline.py` does not load an MLflow
model version or alias.

Therefore:

> MLflow is used as an experiment tracker and artifact viewer, not as the
> authoritative deployment registry.

---

## 10. Inference and Redis caching

The inference path correctly loads the child model and matching child scaler,
uses the latest 60 rows, runs one no-gradient forward pass, inverse-transforms
the five forecast rows, and creates business-day dates.

Redis provides:

- One-day prediction cache.
- Background task state.
- Fixed-window rate limiting.
- Cache hit/miss counters.

### Critical bug: Redis client is bound at import time

`backend/tasks.py` and `backend/api.py` do:

```python
from backend.state import redis_client
```

At import time `backend.state.redis_client` is `None`. Startup later assigns:

```python
app_state.redis_client = client
```

That updates the attribute on `backend.state`, but **not** the already-imported
names in `tasks.py` and `api.py`. Those modules keep `None`.

Rate limiting still works because `backend/rate_limiter.py` imports Redis
inside the request. Prediction cache, task-status persistence, duplicate-job
prevention, `/system/cache`, and Redis reset therefore do not. Logs saying
“Systems online” only prove that startup pinged Redis.

This is the kind of production bug our version must not copy.

### Other cache limitations

- Cache identity contains ticker only, not model version, data timestamp, or
  feature version. Retraining can leave stale cached predictions for a day.
- If Redis access throws, `get_or_set_cache()` can execute `compute_fn()` again,
  potentially duplicating expensive inference or side effects.
- Cache miss is incremented only when Redis exists.
- The API startup hardcodes Redis host `redis` instead of consistently using
  environment settings.

---

## 11. LangGraph agents in detail

### Shared state

`AgentState` extends `MessagesState` with:

- ticker
- predictions
- news sentiment
- final report
- recommendation
- confidence

`MemorySaver` stores graph checkpoints in the API worker's memory. It is not a
durable or shared production checkpoint store. Worse, `build_graph()` creates a
**new graph and new MemorySaver on every analysis request**, so `thread_id`
does not preserve conversation across HTTP calls. It only labels checkpoints
inside that one invocation.

### Performance analyst

The node prompts the LLM to summarize trend and range. Its LLM response is added
to `messages`, but there is no dedicated `performance_analysis` field.

This creates a key issue: the report generator receives raw prediction data but
does not explicitly receive the performance analyst's text except indirectly
through accumulated message state. Its own prompt only inserts `predictions`
and `news`.

### Market expert

`get_stock_news()`:

1. Calls Finnhub for the previous seven days.
2. Uses up to five articles.
3. Falls back to `YahooFinanceNewsTool`.

The LLM generates a sentiment summary, but the node stores the original news
text in `news_sentiment`, not the generated summary. The report generator
therefore sees raw news text.

### Report generator

It asks for Bloomberg-style markdown and extracts stance/confidence through
substring checks.

This parsing is fragile:

- Any occurrence of “bullish” wins before “bearish.”
- “HIGH” can occur outside the confidence field.
- The model is not forced into a structured schema.

A structured Pydantic/JSON output should carry recommendation, confidence,
citations, claims, and evidence.

### Critic

The critic rewrites the draft and replaces `final_report`. It does not update
the previously extracted recommendation or confidence. The final report can
therefore disagree with the response metadata.

There is no conditional edge back to the report generator, no numeric critic
score, and no maximum revision loop. This is one-pass editing, not a
self-correcting reflection cycle.

### Bound tools are not actually selected by the LLM

The Ollama model is bound to `TOOLS_LIST`, but the graph itself calls prediction
and news functions deterministically. There is no LangGraph tool node or
tool-call routing. The system is an orchestrated pipeline of LLM nodes rather
than a tool-selecting autonomous agent.

That is not inherently bad; deterministic workflows are often safer. The
terminology should simply be precise.

---

## 12. Why an LLM may give a wrong answer here

The current system can fail at several layers:

### Bad upstream forecast

The LLM may faithfully explain an inaccurate LSTM result. A fluent explanation
does not validate the forecast.

### Weak evidence contract

The report prompt receives prediction text and news text without stable source
IDs, timestamps, freshness indicators, or structured facts.

### Prompt-only grounding

Nothing checks every numerical claim in the report against the prediction
payload.

### Critic is another generative model call

The critic may introduce new unsupported content. It is not a deterministic
validator.

### Metadata/report disagreement

Recommendation and confidence are extracted before the critic rewrite.

### Semantic cache staleness

A report can remain cached for 24 hours even after new market events or model
retraining.

### Evaluation is superficial

The evaluator checks whether:

- the ticker appears,
- the text contains a digit,
- words such as source/news/data appear,
- a recommendation word appears.

A hallucinated report can pass all three.

---

## 13. How agent evaluation actually works

`src/monitoring/agent_eval.py::AgentEvaluator` runs the full agent and gives
one point each for:

1. Ticker relevance.
2. Apparent trustworthiness based on keywords plus any digit.
3. Presence of a recommendation word.

Scores above 0.6 are labeled “Trustworthy.”

This is a smoke test for output shape, not factual or financial evaluation.

### A serious evaluation suite should add

- Forecast numbers in report exactly match supplied model output.
- Every news claim maps to a source URL and timestamp.
- No fabricated ticker, price, date, company, or event.
- Recommendation direction agrees with a deterministic forecast rule.
- Critic output does not alter factual numbers.
- Structured-output schema validity.
- Groundedness/faithfulness scoring.
- Pairwise regression tests against a known set of ticker scenarios.
- Human review samples.
- Latency, token use, cache hit, and failure rate.

The most useful debugging artifact is a per-run trace containing node input,
prompt, model output, parsed state, source documents, and validation results.

---

## 14. Drift monitoring

The README says Evidently AI, but `src/monitoring/drift.py` explicitly uses a
custom implementation with “No heavy dependencies (Evidently removed).”

For the parent only, it compares:

- Reference: roughly 180 to 30 days ago.
- Current: roughly the latest 30 days.

For each OHLCV column it computes absolute mean shift divided by reference
standard deviation. It also compares close-return volatility.

This is useful as a simple signal, but:

- It is data/covariate drift, not proof of concept drift.
- It does not use actual forecast outcomes.
- It does not monitor model residuals.
- Thresholds are hand-set.
- Child ticker drift is deliberately skipped by the API.
- It uses raw OHLCV levels, so trends can appear as drift.

---

## 15. FastAPI orchestration

`backend/api.py` exposes:

- `GET /` — project metadata.
- `GET /health` — unconditional healthy response.
- `POST /train-parent`.
- `POST /train-child`.
- `POST /predict-parent`.
- `POST /predict-child`.
- `POST /analyze`.
- `GET /status/{task_id}`.
- Monitoring/report endpoints.
- Output and log browsing.
- Redis cache inspection.
- Destructive system reset.

### Background jobs

`backend/tasks.py` uses:

- `asyncio.create_task()`.
- A four-worker `ThreadPoolExecutor`.
- Redis task status with TTL.

This is suitable for a local demonstration, not durable execution. A production
design would use a queue such as Celery/RQ/Arq, Kubernetes Jobs, or a workflow
orchestrator, with idempotency, retries, persisted job state, and resource
isolation.

### Security observations

- CORS allows every origin.
- No authentication or authorization.
- `/system/reset` can erase Redis, Qdrant, Feast data, and all outputs.
- `/system/logs` exposes logs.
- Rate limits are endpoint/ticker based, not reliably client-identity based.
- Health returns 200 without checking Redis, Qdrant, model, Feast, or Ollama.
- Four Uvicorn processes each maintain separate in-memory graph checkpoints,
  globals, and thread pools.

---

## 16. Frontend behavior

`frontend/app.py`:

- Accepts a ticker.
- Calls `/analyze`.
- Polls training status when needed.
- Displays recommendation, confidence, report, and forecast.
- Uses a session UUID as LangGraph thread ID.

The response parsing is defensive because several possible prediction shapes
exist. That is a sign the backend contract is not strict enough.

There is also a likely history-data mismatch: `predict_child()` attaches
`history` beside `predictions`, while `analyze_stock()` injects only the nested
`predictions` object into graph output. The Streamlit chart expects history
inside that object, so historical chart data can be lost in the agent path.

---

## 17. Monitoring application and Prometheus

### Monitoring Streamlit

`monitoring_app/app.py` can:

- Trigger parent training.
- Run parent drift checks.
- Run agent evaluation for a ticker.
- Read evaluation/drift JSON from shared `outputs/`.
- Display recent API logs.

### Prometheus

Custom metrics include:

- CPU, memory, and disk used.
- Redis status/key count.
- Training status/duration/MSE.
- Prediction count/latency.
- Cache hits/misses.
- Automatic FastAPI request metrics.

Gaps:

- `TRAINING_MSE` checks for top-level `"mse"`, but training returns metrics under
  `"metrics"` and names the metric `"MSE"`, so this gauge is unlikely to update.
- Cache key is used as a Prometheus label, which can create high cardinality.
- No model version, data version, drift-alert, LLM token, agent failure, or
  groundedness metrics.
- Grafana default credentials are `admin/admin`.

---

## 18. Docker Compose

`docker-compose.yml` launches:

1. FastAPI.
2. Redis Stack and RedisInsight.
3. Qdrant.
4. Prometheus.
5. Grafana.
6. Main Streamlit frontend.
7. Monitoring Streamlit frontend.

All services share one bridge network. Outputs and feature-store paths are
mounted into FastAPI; outputs are mounted into the monitoring app.

Important details:

- FastAPI uses four workers while also running local background jobs.
- Source code is bind-mounted over the image at `/app`.
- Ollama runs on the host through `host.docker.internal`.
- Redis and Qdrant persist through named volumes.
- There are dependencies but no Compose health conditions.
- Images use floating `latest` tags for infrastructure services.
- No Qdrant/Redis authentication is configured.

---

## 19. Kubernetes and AWS

The `k8s/` directory contains manifests for FastAPI, frontend, monitoring,
Redis, Qdrant, Prometheus, Grafana, and volumes.

They are primarily a Minikube setup:

- App images use local names.
- `imagePullPolicy: Never`.
- FastAPI has one replica.
- Services use `LoadBalancer`.
- Local-style persistent volumes are used.

The FastAPI manifest includes resource requests/limits and probes, but `/health`
is not a readiness check because it always returns healthy.

### What is not present in this clone

- No `.tf` Terraform files.
- No `.github/workflows/` CI/CD workflow.
- No automated tests directory.

`doc/AWS.md` tells the reader to create Terraform and GitHub Actions files. It is
a deployment proposal/tutorial, not proof that this checked-out branch contains
that automation. The repository documentation mentions an AWS deployment
branch; this clone should not be assumed to include it.

---

## 20. Claims versus implementation

### Implemented in meaningful form

- Parent/child LSTM training.
- Freeze or fine-tune transfer strategy.
- Local model/scaler serving.
- FastAPI orchestration.
- Redis rate limiting (working). Prediction cache and task status exist in
  code but are disabled by an import-time Redis client binding.
- Linear LangGraph report workflow.
- Finnhub/Yahoo news.
- Ollama LLM and embeddings.
- Qdrant ticker/TTL report cache.
- Streamlit user and monitoring interfaces.
- Prometheus metrics.
- Docker Compose stack.
- Kubernetes manifests.

### Partially implemented or overstated

- **Feast consistency:** Feast receives data and online reads occur, but training
  and inference use Yahoo dataframes directly.
- **MLflow registry:** runs/artifacts are tracked, but serving uses local files
  and registration is commented out.
- **Redis cache/task status:** intended, but `from backend.state import
  redis_client` captures `None` at import time. Only the rate limiter re-imports
  Redis per request.
- **Auto-healing:** missing models trigger an in-process background task, not a
  durable recovery system. Without working Redis status, parent/child sequencing
  and duplicate-training guards are unsafe.
- **Agent evaluation:** heuristic format checks, not factual evaluation.
- **Concept drift:** volatility/data shift proxy, not outcome-based concept
  drift.
- **Semantic search:** effectively an embedding-based exact-ticker cache.
- **Production readiness:** significant testing, security, data correctness,
  reliability, and deployment gaps remain.

### Documented but absent from this clone

- Terraform implementation.
- GitHub Actions CI/CD.
- Automated test suite.
- Actual Evidently integration.
- Demonstrated 99.9% uptime or stated throughput/latency benchmarks.

---

## 21. What our project should learn from Karan's work

### Adopt the system-level lessons

- Separate user UI, API, cache, vector store, and monitoring concerns.
- Give model training and prediction explicit API contracts.
- Use background jobs for expensive work.
- Preserve agent state as a typed contract.
- Connect forecast output to agent analysis.
- Trace and evaluate every layer, not only the final response.
- Containerize dependencies and provide one-command local startup.
- Make system health visible through metrics and a dashboard.
- Treat deployment manifests and infrastructure as code.

### Improve before adopting

- Use chronological, leakage-safe ML evaluation.
- Compare models to persistence and statistical baselines.
- Predict returns or otherwise use scale-invariant cross-ticker features.
- Promote only models that improve held-out metrics.
- Make Feast the real source for training and serving.
- Use MLflow aliases/versions as the real registry or explicitly choose a local
  artifact registry—do not maintain two authorities.
- Move training from API threads to durable jobs.
- Use structured LLM outputs and deterministic validators.
- Evaluate factual grounding, not merely keywords.
- Version all caches by model/data/prompt versions.
- Add authentication and protect destructive/admin endpoints.
- Add unit, integration, contract, ML, agent, Compose, and deployment tests.

### Do not blindly copy

- Random time-series splitting.
- Fitting scalers on all data.
- Evaluating on training windows.
- Triggering Feast apply/materialization on every Yahoo fetch.
- Assuming a critic LLM guarantees truth.
- Calling a fixed linear graph “autonomous” without qualification.
- Treating docs or manifests as evidence that production deployment works.

---

## 22. Relationship to our `DESIGN.md`

Karan's work validates the overall direction of our design:

```text
ML forecast
  -> serving API
  -> analyst/news/report/critic workflow
  -> Redis/Qdrant memory
  -> UI
  -> monitoring
  -> containers
  -> Kubernetes/cloud
```

Our goal should not be to reproduce every implementation detail. Our version
should preserve this connected journey while strengthening:

- model correctness,
- champion selection,
- model/data lineage,
- LLM grounding and evaluation,
- durable orchestration,
- test coverage,
- security,
- deployability.

Karan's repository is most valuable as a **map of components**. Our repository
should become the more defensible implementation of those components.

---

## 23. Recommended learning sequence

### Pass 1: See the whole request

Read in this order:

1. `frontend/app.py`
2. `backend/api.py`
3. `src/agents/graph.py`
4. `src/agents/tools.py`
5. `src/pipelines/inference_pipeline.py`
6. `src/inference.py`

Goal: explain exactly how one ticker becomes a report.

### Pass 2: Understand model creation

1. `src/config.py`
2. `src/data/ingestion.py`
3. `src/data/preparation.py`
4. `src/model/definition.py`
5. `src/model/training.py`
6. `src/model/evaluation.py`
7. `src/pipelines/training_pipeline.py`

Goal: draw tensor shapes, artifact paths, and all leakage boundaries.

### Pass 3: Understand agents and wrong answers

1. `src/agents/nodes.py`
2. `src/agents/graph.py`
3. `src/memory/semantic_cache.py`
4. `src/monitoring/agent_eval.py`

Goal: identify which state each node receives, what claims are grounded, and
what the evaluator can and cannot catch.

### Pass 4: Understand operations

1. `backend/tasks.py`
2. `backend/state.py`
3. `backend/rate_limiter.py`
4. `docker-compose.yml`
5. `prometheus/prometheus.yml`
6. `k8s/*.yaml`
7. `doc/AWS.md`

Goal: explain local execution, failure recovery, observability, and the gap from
local containers to cloud deployment.

---

## 24. Commands to explore Karan's work safely

Run these from `Stock Agent Ops clone/stock-agent-ops/`.

### Install

```bash
uv sync
```

### Start the local stack

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f fastapi
```

### Inspect API and health

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### Train and predict

```bash
curl -X POST http://localhost:8000/train-parent
curl http://localhost:8000/status/parent

curl -X POST http://localhost:8000/train-child \
  -H "Content-Type: application/json" \
  -d '{"ticker":"NVDA"}'

curl -X POST http://localhost:8000/predict-child \
  -H "Content-Type: application/json" \
  -d '{"ticker":"NVDA"}'
```

### Run the full agent path

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker":"NVDA","thread_id":"learning-session-1"}'
```

### Observe the system

```bash
curl http://localhost:8000/system/cache
curl http://localhost:8000/outputs
curl -X POST http://localhost:8000/monitor/%5EGSPC
curl -X POST http://localhost:8000/monitor/NVDA
```

Local interfaces:

- API docs: `http://localhost:8000/docs`
- Main UI: `http://localhost:8501`
- Monitoring UI: `http://localhost:8502`
- RedisInsight: `http://localhost:8001`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Qdrant: `http://localhost:6333`

Do not execute `DELETE /system/reset` unless you intentionally want to erase
all local models, outputs, caches, and Feast state.

---

## 25. Interview-level explanation

An accurate short explanation is:

> Karan's Stock-Agent-Ops is an end-to-end demo of an agentic MLOps platform.
> A PyTorch LSTM is trained first on the S&P 500 index, then fine-tuned into
> ticker-specific child models. FastAPI orchestrates background training,
> inference, Redis caching, and a linear LangGraph workflow. The agents combine
> model forecasts with recent news, generate a report, and pass it through a
> critic. Qdrant caches reports, Streamlit provides user and monitoring UIs,
> Prometheus/Grafana expose operational metrics, and Docker Compose/Kubernetes
> package the system. The architecture is educationally complete, but its ML
> splits, evaluation, Feast/registry authority, agent grounding, testing, and
> production security need strengthening.

That answer demonstrates both architecture knowledge and engineering judgment.

---

## 26. Questions we should answer while building our version

### ML

- What is the exact prediction target?
- What does persistence predict, and does the LSTM beat it?
- Are train/validation/test windows chronologically isolated?
- Which scaler and feature definitions serve each model?
- Why was a child promoted?
- How do delayed actuals update performance metrics?

### Agents

- Which claims came from forecasts, news, or the LLM?
- Can every number be traced to structured input?
- Can the critic change facts?
- What happens when sources disagree or are stale?
- Which test catches an incorrect recommendation?

### MLOps

- What is the authoritative model registry?
- What is the authoritative feature source?
- Which versions identify model, features, data, prompts, and LLM?
- How is a failed training job retried safely?
- How does rollback work?

### Deployment

- What persists after a pod restart?
- Which secrets and endpoints are protected?
- What does readiness actually verify?
- Can multiple replicas safely train, materialize Feast, and write artifacts?
- What alerts tell us predictions or reports are degrading?

These questions turn the project from a collection of tools into an explainable
engineering system.

---

## 27. Additional code-level findings from the ML-path review

These details refine the sections above. They come from reading the training,
inference, Feast, and serving code rather than the README.

### Training and evaluation

- `Config` duplicates `fine_tune_lr`, `parent_dir`, and `workdir`. Harmless in
  Python dataclasses, but a sign the config is not a single source of truth.
- Sequence construction uses `range(context_len, len(df) - pred_len)`, which
  skips one otherwise-valid final window.
- Frozen LSTM layers still run with `model.train()`, so dropout remains active
  on frozen representations.
- `compare_transfer_learning.py` loads parent weights but then trains every
  layer at `1e-3`. It does **not** evaluate the freeze/fine-tune strategy used
  by `train_child()`.
- `TRAINING_MSE` looks for a top-level `"mse"` field, while training returns
  metrics under `"metrics"` as `"MSE"`. The Prometheus gauge is unlikely to
  update.

### Artifacts and serving

- Child models are saved under uppercase directories (`outputs/AAPL/...`)
  because the API uppercases tickers. Several inspection/monitor routes then
  look under lowercase paths (`outputs/aapl/...`). That works on Windows and
  fails on Linux containers.
- Shared Feast parquet `drop_duplicates()` keeps the **first** row, so later
  refreshed market data does not replace older values.
- Docker installs unpinned `backend/requirements.txt` on Python 3.13, while
  local `uv.lock` pins an old Feast (`0.20.0`). Local and container stacks can
  diverge.
- This clone has empty `outputs/`, no Feast parquet, and no completed MLflow
  runs. The architecture can be studied; successful training is not evidenced
  in the checkout.

### What this means for us

When we implement the same layers, we should:

1. Bind Redis through a getter or `backend.state.redis_client` lookups, never
   a one-time imported `None`.
2. Keep artifact paths case-consistent.
3. Make Feast the actual train/serve source, or drop the consistency claim.
4. Treat MLflow as tracker **or** registry, not both with only one used.
5. Never use random overlapping-window splits for time series.

---

## 28. Additional findings from the agent/API review

These refine the request-to-report and UI sections.

### Auto-training and UI polling

- `fetch_prediction_data()` turns every HTTP `202` into `"__MODEL_TRAINING__"`
  and **drops the task ID**. Streamlit then polls `/status/{ticker}`.
- That accidentally works when only a **child** model is missing, because the
  child task ID is the lowercase ticker.
- If the **parent** is also missing, the API returns task ID `parent_training`.
  The UI still polls the child ticker. Child training is not started after
  parent completion. The spinner can run forever.
- Because Redis task status is disabled, `/status/{ticker}` is 404 until a
  model file appears on disk. Failed training with no artifact has no timeout.
- `get_or_set_cache()` catches cache **and** compute errors together, then
  calls `compute_fn()` again. A failed inference can run twice.

### Contracts and frontend

- `/analyze` is not rate-limited, despite docs listing 40/hour.
- Training and error cases return HTTP **200** with `status: training|error`.
  Clients must inspect the body. Only thrown exceptions become 500.
- Fresh analysis returns the full graph state (including `messages`). A Qdrant
  cache hit returns a smaller DTO. Those shapes differ.
- `predict_child()` attaches `history` **beside** `predictions`. `analyze_stock()`
  then injects only `result.predictions` into the graph output, so Streamlit
  usually has **no historical series**. “Latest price” falls back to the first
  forecast.
- Qdrant `last_price` looks for `historical`, while inference uses `history`,
  so cached last-price also falls back to the first forecast.
- `AnalyzeRequest.use_fmi` is never read.

### Agents

- Bound tools (`get_stock_predictions`, `get_stock_news`) have no LangGraph
  ToolNode. If the LLM emits a tool call, nothing handles it.
- If Ollama init fails, the mock `invoke` closes over exception `e`. After the
  `except` block, `e` is cleared, so the mock can raise `NameError`.
- `GOOGLE_API_KEY` / `langchain-google-genai` appear in docs and dependencies,
  but only Ollama is implemented.

### Ops for the agent path

- Four Uvicorn workers each own a Prometheus registry. A scrape hits one
  worker; counters are incomplete.
- Kubernetes FastAPI has no `OLLAMA_BASE_URL`, so the default `localhost:11434`
  is inside the pod.
- `k8s/monitoring-app.yaml` does not mount the outputs PVC. Drift/eval JSON
  written by FastAPI will not appear in the monitoring UI on Kubernetes.
- Grafana Compose has admin/admin; Kubernetes Grafana is not provisioned with
  a Prometheus datasource.

### What this means for us

6. Return a **single** analyze DTO. Never leak graph internals or diverge
   cache-hit vs cache-miss shapes.
7. Pass `task_id` through training responses and poll **that** ID, with a
   deadline.
8. Put history inside the prediction contract the UI actually reads.
9. Recompute recommendation/confidence **after** the critic, from structured
   output—not substring search on a draft.
10. Recreate MemorySaver only if you intend a new conversation; otherwise use
    a durable checkpointer.
