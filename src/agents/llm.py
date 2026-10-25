"""Shared chat LLM client — Ollama or Google via LLM_PROVIDER."""

from __future__ import annotations

import logging
import os
import re
import time
import warnings

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from logger.logger import get_logger, log_event

load_dotenv()
logger = get_logger("sip.llm")


def _mock_llm(reason: str):
    class _MockLLM:
        def invoke(self, messages):
            return AIMessage(
                content=(
                    f"Mock LLM unavailable ({reason}). "
                    "Market Stance: NEUTRAL | Confidence: Low"
                )
            )

    return _MockLLM()


def _ollama_llm():
    from langchain_ollama import ChatOllama

    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "900"))
    log_event(
        logger,
        "llm_client_ready",
        step="llm_init",
        status="ok",
        data={"provider": "ollama", "model": model, "base_url": base_url},
    )
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
        num_predict=num_predict,
    )


def _messages_to_parts(messages: list[BaseMessage] | list) -> tuple[str | None, str]:
    """Split LangChain messages into optional system instruction + user text."""
    system_chunks: list[str] = []
    user_chunks: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        if content is None:
            text = str(message)
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            text = "\n".join(parts)
        else:
            text = str(content)

        if isinstance(message, SystemMessage):
            system_chunks.append(text)
        elif isinstance(message, HumanMessage):
            user_chunks.append(text)
        elif isinstance(message, AIMessage):
            user_chunks.append(f"Assistant: {text}")
        else:
            # Fallback for dict-like / unknown roles
            role = getattr(message, "type", "") or ""
            if str(role).lower() == "system":
                system_chunks.append(text)
            else:
                user_chunks.append(text)

    system = "\n\n".join(system_chunks).strip() or None
    user = "\n\n".join(user_chunks).strip()
    if not user:
        user = system or ""
        system = None
    return system, user


def _is_gemma_model(model: str) -> bool:
    return (model or "").strip().lower().startswith("gemma")


def _is_transient_llm_error(exc: BaseException) -> bool:
    """Google / network blips that often succeed on a short backoff retry."""
    text = f"{type(exc).__name__} {exc}".lower()
    needles = (
        "500",
        "503",
        "429",
        "internal",
        "unavailable",
        "resource_exhausted",
        "resource exhausted",
        "deadline",
        "timed out",
        "timeout",
        "temporarily",
        "connection reset",
        "connection aborted",
        "broken pipe",
        "server error",
        "internal error encountered",
    )
    return any(n in text for n in needles)


class GoogleChatLLM:
    """Google GenAI via models.generate_content (works for Gemini + hosted Gemma)."""

    # Burst runs (ticker 2/3 after a success) often hit Google 500 INTERNAL.
    _MAX_ATTEMPTS = 4
    _BACKOFF_SEC = (0.6, 1.2, 2.4)

    def __init__(self, *, api_key: str, model: str, temperature: float) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self._model = model
        self._temperature = temperature
        self._client = genai.Client(api_key=api_key)
        self._is_gemma = _is_gemma_model(model)

    def _build_config(self, system: str | None, *, use_thinking: bool):
        config_kwargs: dict = {"temperature": self._temperature}
        if system and not self._is_gemma:
            config_kwargs["system_instruction"] = system
        if self._is_gemma and use_thinking:
            try:
                config_kwargs["thinking_config"] = self._types.ThinkingConfig(
                    thinking_level="minimal"
                )
            except Exception:  # noqa: BLE001
                pass
        return self._types.GenerateContentConfig(**config_kwargs)

    def invoke(self, messages: list) -> AIMessage:
        system, user = _messages_to_parts(messages)
        # Gemma hosted docs use generate_content; fold system into the user turn
        # if system_instruction is flaky for a given model build.
        if system and self._is_gemma:
            user = f"SYSTEM INSTRUCTIONS:\n{system}\n\nUSER:\n{user}"
            system_for_config = None
        else:
            system_for_config = system

        started = time.perf_counter()
        warnings.filterwarnings(
            "ignore",
            message=r".*automatic function calling \(AFC\).*",
            category=UserWarning,
        )

        last_error: BaseException | None = None
        response = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            # After a Gemma failure, drop thinking_config — it is a known 500 trigger.
            use_thinking = self._is_gemma and attempt == 1
            config = self._build_config(system_for_config, use_thinking=use_thinking)
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user,
                    config=config,
                )
                if attempt > 1:
                    log_event(
                        logger,
                        "llm_transient_retry_ok",
                        level=logging.WARNING,
                        step="llm_call",
                        status="ok",
                        data={
                            "model": self._model,
                            "attempt": attempt,
                            "prior_error": str(last_error)[:240] if last_error else "",
                        },
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                transient = _is_transient_llm_error(exc)
                will_retry = attempt < self._MAX_ATTEMPTS and (
                    transient or (self._is_gemma and attempt == 1)
                )
                log_event(
                    logger,
                    "llm_invoke_retry" if will_retry else "llm_invoke_failed",
                    level=logging.WARNING,
                    step="llm_call",
                    status="retry" if will_retry else "error",
                    data={
                        "model": self._model,
                        "attempt": attempt,
                        "transient": transient,
                        "error": str(exc)[:320],
                    },
                )
                if not will_retry:
                    raise
                time.sleep(self._BACKOFF_SEC[min(attempt - 1, len(self._BACKOFF_SEC) - 1)])

        if response is None:
            assert last_error is not None
            raise last_error

        duration_ms = (time.perf_counter() - started) * 1000
        text = getattr(response, "text", None) or str(response)
        log_event(
            logger,
            "llm_invoke",
            step="llm_call",
            status="ok",
            duration_ms=duration_ms,
            metrics={"chars": len(text or ""), "provider": "google"},
            data={"model": self._model, "gemma": self._is_gemma},
        )
        return AIMessage(content=text)


def _google_llm():
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "LLM_PROVIDER=google but GOOGLE_API_KEY is empty in .env"
        )
    model = os.getenv("GOOGLE_MODEL", "gemini-3.8-flash")
    temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0.3"))

    try:
        client = GoogleChatLLM(
            api_key=api_key, model=model, temperature=temperature
        )
        log_event(
            logger,
            "llm_client_ready",
            step="llm_init",
            status="ok",
            data={
                "provider": "google",
                "model": model,
                "transport": "models.generate_content",
                "gemma": _is_gemma_model(model),
            },
        )
        return client
    except Exception as primary:  # noqa: BLE001
        warnings.filterwarnings(
            "ignore",
            message=r".*automatic function calling \(AFC\).*",
            category=UserWarning,
        )
        from langchain_google_genai import ChatGoogleGenerativeAI

        log_event(
            logger,
            "llm_client_fallback",
            step="llm_init",
            status="degraded",
            data={
                "provider": "google",
                "model": model,
                "transport": "langchain_generate_content",
                "error": str(primary),
            },
        )
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )


def get_chat_llm():
    """Return chat model for agents. Switch with LLM_PROVIDER=ollama|google."""
    provider = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    try:
        if provider in {"google", "gemma", "gemini"}:
            return _google_llm()
        if provider in {"ollama", "local"}:
            return _ollama_llm()
        raise RuntimeError(
            f"Unknown LLM_PROVIDER={provider!r}; use 'ollama' or 'google'"
        )
    except Exception as exc:  # noqa: BLE001 - keep graph runnable offline
        log_event(
            logger,
            "llm_init_failed",
            level=logging.WARNING,
            step="llm_init",
            status="error",
            data={"provider": provider, "error": str(exc)},
        )
        return _mock_llm(str(exc))


def message_text(response) -> str:
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            text = "\n".join(parts)
        else:
            text = str(content)
    else:
        text = str(response)
    return re.sub(r"(?i)^\s*assistant\s*\n+", "", text).strip()
