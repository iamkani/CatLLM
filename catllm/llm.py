# catllm/llm.py
import os
from typing import List, Dict, Optional, Tuple, Any

from openai import OpenAI, AzureOpenAI

def ensure_client(provider: str):
    """
    Returns a correctly configured OpenAI client for the given provider.
    - provider: "OpenAI" or "Azure OpenAI"
    """
    if provider == "Azure OpenAI":
        key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        if not key or not endpoint:
            raise RuntimeError("Azure OpenAI: AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set.")
        # AzureOpenAI client requires azure_endpoint + api_version
        return AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)

    # Default: OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI: OPENAI_API_KEY is not set in this environment.")
    return OpenAI(api_key=api_key)


# catllm/llm.py
def synthesize_answer(client, provider: str, chat_model: str, messages: list[str]) -> str:
    # For OpenAI: chat_model is like "gpt-4o-mini"
    # For Azure: chat_model is the *deployment name* you created
    resp = client.chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0.1,
    )
    return resp.choices[0].message.content
