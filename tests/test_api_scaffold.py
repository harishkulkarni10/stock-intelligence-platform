from fastapi.testclient import TestClient

from backend import main
from backend.main import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["project"] == "Stock Intelligence Platform"


def test_ready():
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert "redis" in body["details"]


def test_ready_requires_models_only_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setenv("REQUIRE_MODELS", "true")

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "not_ready"
