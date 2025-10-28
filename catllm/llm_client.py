# catllm/llm_client.py
from __future__ import annotations

import os
import math
import time
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from openai import OpenAI, AzureOpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore

from .tagging import role_hint
from .utils_text import approx_token_count

logger = logging.getLogger(__name__)

try:
    from .utils_text import approx_token_count  # preferred
except Exception:
    def approx_token_count(s: str) -> int:  # fallback
        return max(1, len(s or "") // 4)
# -----------------------------
# Client construction
# -----------------------------
def ensure_client(provider: str):
    """
    Return an OpenAI or Azure OpenAI client using environment variables.

    OpenAI:
      - OPENAI_API_KEY
    Azure:
      - AZURE_OPENAI_KEY
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_VERSION (default: 2024-08-01-preview)
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
    else:
        raise RuntimeError(f"Unsupported provider: {provider}")


# -----------------------------
# Embeddings (safe-batched)
# -----------------------------
def embed_texts(
    client,
    texts: Sequence[str],
    embed_model: str,
    max_req_tokens: int = 7000,
    max_items: int = 32,
    sleep_between: float = 0.1,
) -> np.ndarray:
    """
    Create embeddings for a list of texts with token-safe batching.
    Heuristic: 4 chars ≈ 1 token.
    """
    if not texts:
        return np.zeros((0, 1), dtype="float32")

    embs: List[List[float]] = []
    i = 0
    while i < len(texts):
        cur: List[str] = []
        tok_sum = 0
        while i < len(texts) and len(cur) < max_items:
            tks = approx_token_count(texts[i])
            if cur and tok_sum + tks > max_req_tokens:
                break
            cur.append(texts[i])
            tok_sum += tks
            i += 1

        resp = client.embeddings.create(model=embed_model, input=cur)
        embs.extend([d.embedding for d in resp.data])
        if sleep_between > 0:
            time.sleep(sleep_between)

    return np.array(embs, dtype="float32")


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