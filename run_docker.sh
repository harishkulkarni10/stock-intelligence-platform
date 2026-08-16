#!/usr/bin/env bash
set -euo pipefail
docker compose up --build -d
echo "API http://localhost:8000/docs"
echo "Grafana http://localhost:3000"
echo "Prometheus http://localhost:9090"
