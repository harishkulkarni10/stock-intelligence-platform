import time

from fastapi.testclient import TestClient

from backend import api, state, tasks
from backend.main import app
from backend.rate_limiter import FixedWindowRateLimiter


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.counts = {}

    def setex(self, key, _ttl, value):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)

    def incr(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, _key, _ttl):
        return True


def test_task_status_keeps_exact_id_and_result(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(state, "get_redis", lambda: fake)
    task_id = tasks.submit_task("unit", lambda: {"done": True})

    for _ in range(100):
        status = tasks.get_task_status(task_id)
        if status and status["status"] == "completed":
            break
        time.sleep(0.01)

    assert status["task_id"] == task_id
    assert status["result"] == {"done": True}


def test_task_status_records_errors(monkeypatch):
    monkeypatch.setattr(state, "get_redis", lambda: None)

    def fail():
        raise RuntimeError("training failed")

    task_id = tasks.submit_task("unit-failure", fail)
    for _ in range(100):
        status = tasks.get_task_status(task_id)
        if status and status["status"] == "failed":
            break
        time.sleep(0.01)

    assert status["task_id"] == task_id
    assert status["error"] == "training failed"


def test_prediction_cache_uses_model_version(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(state, "get_redis", lambda: fake)
    result = {
        "ticker": "NVDA",
        "horizon": 5,
        "model_version": "v7",
        "predictions": [{"step": 1, "value": 10.0}],
    }

    tasks.cache_prediction("child", result)

    assert "prediction:child:NVDA:5:v7" in fake.values
    assert tasks.get_cached_prediction("child", "NVDA", 5) == result


def test_cache_and_rate_limit_fail_open(monkeypatch):
    class BrokenRedis:
        def get(self, _key):
            raise ConnectionError

        def incr(self, _key):
            raise ConnectionError

    monkeypatch.setattr(state, "get_redis", lambda: BrokenRedis())

    assert tasks.get_cached_prediction("parent", "SPY", 5) is None
    assert FixedWindowRateLimiter().is_allowed("predict:SPY", 1, 60)


def test_missing_parent_prediction_returns_exact_task_id(monkeypatch):
    monkeypatch.setattr(api, "_parent_model_exists", lambda: False)
    monkeypatch.setattr(tasks, "submit_task", lambda *_args: "task-exact-123")

    response = TestClient(app).post(
        "/predict-parent", json={"ticker": "SPY", "horizon": 3}
    )

    assert response.status_code == 202
    assert response.json()["task_id"] == "task-exact-123"


def test_child_training_preserves_parent_first(monkeypatch):
    calls = []
    monkeypatch.setattr(tasks, "train_parent", lambda: calls.append("parent"))
    monkeypatch.setattr(tasks, "train_child", lambda ticker: calls.append(ticker))

    tasks.train_child_after_parent("NVDA")

    assert calls == ["parent", "NVDA"]


def test_status_route_not_found():
    response = TestClient(app).get("/status/does-not-exist")
    assert response.status_code == 404
