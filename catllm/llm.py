# catllm/llm.py
from __future__ import annotations

import os
from typing import Any, Dict, Generator, List, Optional, Tuple

from openai import OpenAI, AzureOpenAI

def ensure_client(provider: str, base_url: str = ""):
    """
    Returns a correctly configured OpenAI client for the given provider.

    Providers: "OpenAI", "Azure OpenAI", or "Local (OpenAI-compatible)".
    For Local, supply *base_url* (or set LOCAL_API_BASE env var) pointing at
    any OpenAI-compatible server (vLLM, Ollama, LM Studio, llama.cpp, etc.).
    """
    if provider == "Azure OpenAI":
        key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        if not key or not endpoint:
            raise RuntimeError("Azure OpenAI: AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set.")
        return AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)

    if provider == "Local (OpenAI-compatible)":
        url = base_url or os.getenv("LOCAL_API_BASE", "http://localhost:11434/v1")
        key = os.getenv("LOCAL_API_KEY", "local")
        return OpenAI(api_key=key, base_url=url)

    # Default: OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI: OPENAI_API_KEY is not set in this environment.")
    return OpenAI(api_key=api_key)


def synthesize_answer(client, provider: str, chat_model: str, messages: list[str]) -> str:
    resp = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content


def synthesize_answer_stream(
    client, provider: str, chat_model: str, messages: list[str]
) -> Generator[str, None, None]:
    """Streaming variant — yields text chunks as they arrive."""
    stream = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
