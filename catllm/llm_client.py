# catllm/llm_client.py
from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from openai import OpenAI, AzureOpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore

from .tagging import role_hint

logger = logging.getLogger(__name__)
# -----------------------------
# Client construction
# -----------------------------
def ensure_client(provider: str, base_url: str = ""):
    """
    Return an OpenAI, Azure OpenAI, or Local (OpenAI-compatible) client.
    """
    provider = (provider or "").strip().lower()
    if provider == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package not available.")
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        return OpenAI(api_key=key)
    elif provider in ("azure", "azure openai", "azure-openai"):
        if AzureOpenAI is None:
            raise RuntimeError("openai package (AzureOpenAI) not available.")
        key = os.getenv("AZURE_OPENAI_KEY", "")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        if not key or not endpoint:
            raise RuntimeError("Azure env vars missing: AZURE_OPENAI_KEY/AZURE_OPENAI_ENDPOINT.")
        return AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)
    elif provider in ("local", "local (openai-compatible)"):
        url = base_url or os.getenv("LOCAL_API_BASE", "http://localhost:11434/v1")
        key = os.getenv("LOCAL_API_KEY", "local")
        return OpenAI(api_key=key, base_url=url)
    else:
        raise RuntimeError(f"Unsupported provider: {provider}")


# -----------------------------
# Embeddings — delegated to catllm.embeddings (single source of truth)
# -----------------------------
from .embeddings import embed_texts  # noqa: F401


# -----------------------------
# Role-aware system prompt
# -----------------------------
BASE_SYSTEM_PROMPT = (
    "You are a cattle-genetics RAG assistant.\n"
    "- Use ONLY the provided CONTEXT to answer.\n"
    "- If the answer is not in the context, say you don't know.\n"
    "- Be concise and precise; expand acronyms on first use.\n"
    "- Include inline citations like [#] referencing the context items.\n"
    "- Add a short 'Sources' section listing the cited items with their titles.\n"
    "- For medical or veterinary advice, add: 'This is not veterinary advice.'"
)


def build_system_prompt_for_role(user_role: str) -> str:
    """Augment the base system prompt with role-specific guidance."""
    hint = role_hint(user_role or "")
    return f"{BASE_SYSTEM_PROMPT}\n\nRole guidance for '{user_role or 'Independent Rancher'}': {hint}"


# -----------------------------
# Chat completion
# -----------------------------
def synthesize_answer(
    client,
    provider: str,
    chat_model: str,
    question: str,
    ctx_text: str,
    user_role: str = "Independent Rancher",
    temperature: float = 0.1,
) -> str:
    """
    Compose a role-aware prompt and get an answer from the model.
    """
    sys_prompt = build_system_prompt_for_role(user_role)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx_text}"},
    ]
    resp = client.chat.completions.create(
        model=chat_model, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content


# -----------------------------
# Optional: OpenAI web search fallback
# -----------------------------
def answer_with_openai_web_search(
    client,
    chat_model: str,
    question: str,
    user_role: str = "Independent Rancher",
    temperature: float = 0.1,
) -> Optional[str]:
    """
    Try to use OpenAI's Responses API with the 'web_search' tool (if enabled on your account).
    If unavailable, returns None and your app can handle a graceful fallback.

    Note: This feature may not be available to all orgs or models yet.
    """
    try:
        # Some accounts expose client.responses and the 'web_search' tool.
        sys_prompt = (
            "You are a cattle-genetics assistant. Do a brief web search to gather 2-5 authoritative sources, "
            "then write a concise, role-appropriate answer. Always list the sources (with titles & URLs). "
            "If information is uncertain, state limitations clearly."
        )
        # The exact API surface can vary. We attempt a generic call and catch errors.
        resp = client.responses.create(
            model=chat_model,
            input=[
                {"role": "system", "content": sys_prompt + f"\n\nRole hint: {role_hint(user_role)}"},
                {"role": "user", "content": question},
            ],
            tools=[{"type": "web_search"}],
            temperature=temperature,
        )
        # Common shape: resp.output_text or in choices
        text = getattr(resp, "output_text", None)
        if text:
            return text
        # Fallback parse
        if hasattr(resp, "output") and resp.output and isinstance(resp.output, list):
            for part in resp.output:
                if isinstance(part, dict) and part.get("type") == "message":
                    msg = part.get("content")
                    if isinstance(msg, str) and msg.strip():
                        return msg
        # Some variants pack content in choices
        if hasattr(resp, "choices") and resp.choices:
            c = resp.choices[0]
            if hasattr(c, "message") and getattr(c.message, "content", None):
                return c.message.content
        return None
    except Exception as e:
        logger.debug(f"OpenAI web_search tool not available or failed: {e}")
        return None