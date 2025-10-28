# catllm/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    # Optional: load .env automatically if present
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=False)
except Exception:
    pass


@dataclass(frozen=True)
class Models:
    """Central place for model names (easy to override via env)."""
    chat: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embed: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")


@dataclass(frozen=True)
class Provider:
    """Which LLM provider to use and (optional) Azure settings."""
    name: str = os.getenv("PROVIDER", "OpenAI")  # "OpenAI" | "Azure OpenAI"
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    azure_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    azure_embed_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "")


@dataclass(frozen=True)
class Storage:
    """Paths used by the app."""
    store_dir: str = os.getenv("STORE_DIR", ".rag_store")


@dataclass(frozen=True)
class IngestDefaults:
    """Defaults for chunking and ingest limits; Streamlit can override."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    max_files: int = int(os.getenv("MAX_FILES", "200"))
    max_size_mb: int = int(os.getenv("MAX_SIZE_MB", "50"))
    recursive: bool = os.getenv("RECURSIVE", "true").lower() != "false"
    exts: tuple[str, ...] = tuple(
        os.getenv("INGEST_EXTS", "pdf,txt,md,csv").replace(" ", "").split(",")
    )


@dataclass(frozen=True)
class RetrievalDefaults:
    """Search behavior defaults."""
    top_k: int = int(os.getenv("TOP_K", "5"))
    diversify: bool = os.getenv("DIVERSIFY_RESULTS", "true").lower() != "false"


@dataclass(frozen=True)
class AppFlags:
    """Feature toggles and UI behavior."""
    autosave: bool = os.getenv("AUTOSAVE", "true").lower() != "false"
    use_ocr_default: bool = os.getenv("USE_OCR_DEFAULT", "false").lower() == "true"
    ocr_max_pages: int = int(os.getenv("OCR_MAX_PAGES", "10"))
    ocr_dpi: int = int(os.getenv("OCR_DPI", "200"))
    expand_queries_default: bool = os.getenv("EXPAND_QUERIES", "true").lower() != "false"
    show_expanded_default: bool = os.getenv("SHOW_EXPANDED", "false").lower() == "true"


# Singletons to import elsewhere
MODELS = Models()
PROVIDER = Provider()
STORAGE = Storage()
INGEST = IngestDefaults()
RETRIEVAL = RetrievalDefaults()
FLAGS = AppFlags()