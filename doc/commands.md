# Commands

## Local API

```bash
pip install -e ".[dev]"
uvicorn backend.main:app --reload --port 8000
```

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/metrics
```

## Compose

```bash
copy .env.example .env
docker compose up --build -d
docker compose logs -f fastapi
docker compose down
```

## Planned (not implemented yet)

```bash
curl -X POST http://localhost:8000/train-parent
curl -X POST http://localhost:8000/train-child -H "Content-Type: application/json" -d "{\"ticker\":\"NVDA\"}"
curl -X POST http://localhost:8000/predict-child -H "Content-Type: application/json" -d "{\"ticker\":\"NVDA\"}"
curl http://localhost:8000/status/nvda
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d "{\"ticker\":\"NVDA\"}"
```
