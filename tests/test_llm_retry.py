"""Transient Google LLM error detection / retry helpers."""

from src.agents.llm import _is_transient_llm_error


def test_transient_500_internal():
    err = Exception(
        "500 INTERNAL. {'error': {'code': 500, "
        "'message': 'Internal error encountered.', 'status': 'INTERNAL'}}"
    )
    assert _is_transient_llm_error(err) is True


def test_transient_429_and_timeout():
    assert _is_transient_llm_error(RuntimeError("429 RESOURCE_EXHAUSTED")) is True
    assert _is_transient_llm_error(TimeoutError("deadline exceeded")) is True


def test_non_transient_stays_false():
    assert _is_transient_llm_error(ValueError("invalid argument: bad schema")) is False
