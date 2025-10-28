# catllm/__init__.py
# Keep the package surface minimal; avoid importing heavy modules here.

from .llm import ensure_client, synthesize_answer
from .utils_text import (
    expand_query,
    chunk_text,
    read_docs,
    read_local_folder,
    ocr_pdf_bytes,
)
from .ingest import ingest_uploads, ingest_folder
from .persistence import save_store, load_store
from .indexer import build_faiss_index, top_k_search

# format_context may live in retrieval; it's ok if some UIs provide their own
try:
    from .retrieval import format_context  # noqa: F401
except Exception:
    format_context = None  # optional export
from .prompts import build_system_prompt_for_role, ROLE_TEMPLATES

# Roles helpers (explicit role handling)
from .roles import apply_role_style, normalize_role, UserRole

__all__ = [
    "ensure_client",
    "synthesize_answer",
    "expand_query",
    "chunk_text",
    "read_docs",
    "read_local_folder",
    "ocr_pdf_bytes",
    "ingest_uploads",
    "ingest_folder",
    "save_store",
    "load_store",
    "build_faiss_index",
    "top_k_search",
    "format_context",
    "build_system_prompt_for_role",
    "ROLE_TEMPLATES",
    "apply_role_style",
    "normalize_role",
    "UserRole",
]