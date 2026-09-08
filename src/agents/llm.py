"""Shared Ollama chat client for agent nodes."""

from __future__ import annotations

import os
import re

from langchain_core.messages import AIMessage


def get_chat_llm():
    """Return ChatOllama, or a mock that keeps the graph runnable offline."""
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature, base_url=base_url)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully for local smoke tests
        error = exc

        class _MockLLM:
            def invoke(self, messages):
                return AIMessage(
                    content=(
                        "Mock LLM unavailable "
                        f"({error}). Market Stance: NEUTRAL | Confidence: Low"
                    )
                )

        return _MockLLM()


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
    # Some chat wrappers prefix a role label; strip for clean agent output.
    return re.sub(r"(?i)^\s*assistant\s*\n+", "", text).strip()
