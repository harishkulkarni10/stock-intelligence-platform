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


class GoogleChatLLM:
    """Google GenAI chat via chats.send_message (avoids Models.generate_content AFC warning)."""

    def __init__(self, *, api_key: str, model: str, temperature: float) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self._model = model
        self._temperature = temperature
        self._client = genai.Client(api_key=api_key)

    def invoke(self, messages: list) -> AIMessage:
        system, user = _messages_to_parts(messages)
        config_kwargs: dict = {"temperature": self._temperature}
        if system:
            config_kwargs["system_instruction"] = system
        config = self._types.GenerateContentConfig(**config_kwargs)

        started = time.perf_counter()
        chat = self._client.chats.create(model=self._model, config=config)
        response = chat.send_message(user)
        duration_ms = (time.perf_counter() - started) * 1000
        text = getattr(response, "text", None) or str(response)
        log_event(
            logger,
            "llm_invoke",
            step="llm_call",
            status="ok",
            duration_ms=duration_ms,
            metrics={"chars": len(text or ""), "provider": "google"},
            data={"model": self._model},
        )
        return AIMessage(content=text)


def _google_llm():
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "LLM_PROVIDER=google but GOOGLE_API_KEY is empty in .env"
        )
    model = os.getenv("GOOGLE_MODEL", "gemma-3-12b-it")
    temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0.3"))

    # Prefer Chat.send_message path; fall back to LangChain with warning filter.
    try:
        client = GoogleChatLLM(
            api_key=api_key, model=model, temperature=temperature
        )
        log_event(
            logger,
            "llm_client_ready",
            step="llm_init",
            status="ok",
            data={"provider": "google", "model": model, "transport": "chats.send_message"},
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
