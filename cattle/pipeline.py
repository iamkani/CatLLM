# catllm/pipeline.py
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import faiss  # type: ignore

from .utils_text import (
    read_local_folder,
    read_docs,
    save_store,
    load_store,
)
from .retrieval import (
    build_faiss_index,
    make_chunks_from_docs,
    retrieve_context_for_query,
)
from .llm_client import (
    ensure_client,
    embed_texts,
    synthesize_answer,
    answer_with_openai_web_search,
)


# -----------------------------
# Ingestion
# -----------------------------
def ingest_folder_to_chunks(
    folder_path: str,
    *,
    recursive: bool = True,
    exts: Optional[Sequence[str]] = None,
    max_files: int = 200,
    max_size_mb: int = 50,
    use_ocr: bool = False,
    ocr_max_pages: int = 10,
    ocr_dpi: int = 200,
    chunk_size: int = 900,
    overlap: int = 120,
    progress_cb=None,
) -> Tuple[List[Dict], List[str]]:
    """
    Read local folder (with dedupe + optional OCR) and convert to chunk dicts.
    Returns (chunks, fails).
    """
    docs, fails = read_local_folder(
        folder_path=folder_path,
        exts=exts or ["pdf", "txt", "md", "csv"],
        recursive=recursive,
        max_files=max_files,
        max_size_mb=max_size_mb,
        use_ocr=use_ocr,
        ocr_max_pages=ocr_max_pages,
        ocr_dpi=ocr_dpi,
        progress_cb=progress_cb,
    )
    chunks = make_chunks_from_docs(docs, chunk_size=chunk_size, overlap=overlap)
    return chunks, fails


def ingest_uploads_to_chunks(
    uploads,
    *,
    use_ocr: bool = False,
    ocr_max_pages: int = 10,
    ocr_dpi: int = 200,
    chunk_size: int = 900,
    overlap: int = 120,
) -> List[Dict]:
    """
    Convert Streamlit UploadedFile list to chunk dicts.
    """
    docs = read_docs(
        uploads,
        use_ocr=use_ocr,
        ocr_max_pages=ocr_max_pages,
        ocr_dpi=ocr_dpi,
    )
    # Decorate docs with minimal meta
    docs_norm = [(title, text, {"source": title}) for (title, text) in docs]
    return make_chunks_from_docs(docs_norm, chunk_size=chunk_size, overlap=overlap)


# -----------------------------
# Indexing & persistence
# -----------------------------
def build_index_from_chunks(
    *,
    provider: str,
    chat_model: str,        # not used here, but kept for symmetry
    embed_model: str,
    corpus_chunks: Sequence[Dict],
) -> Tuple[np.ndarray, faiss.Index]:
    """
    Embed corpus chunks and return (embeddings, faiss_index).
    """
    client = ensure_client(provider)
    texts = [c["text"] for c in corpus_chunks]
    embs = embed_texts(client, texts=texts, embed_model=embed_model)
    index = build_faiss_index(embs)
    return embs, index


def save_pipeline_store(
    store_dir: str,
    corpus_chunks: Sequence[Dict],
    embeddings: np.ndarray,
    index: faiss.Index,
) -> None:
    save_store(store_dir, corpus_chunks, embeddings, index)


def load_pipeline_store(
    store_dir: str,
) -> Tuple[Optional[List[Dict]], Optional[np.ndarray], Optional[faiss.Index]]:
    return load_store(store_dir)


# -----------------------------
# Retrieve & answer
# -----------------------------
def generate_answer_from_rag(
    *,
    provider: str,
    chat_model: str,
    embed_model: str,
    index: faiss.Index,
    corpus_chunks: Sequence[Dict],
    user_query: str,
    user_role: str = "Independent Rancher",
    use_synonyms: bool = True,
    top_k: int = 5,
    selected_clusters: Optional[Sequence[str]] = None,
    selected_breeds: Optional[Sequence[str]] = None,
    selected_traits: Optional[Sequence[str]] = None,
    selected_genes: Optional[Sequence[str]] = None,
    allow_openai_web_fallback: bool = False,
) -> Tuple[str, List[Dict]]:
    """
    Retrieve context from FAISS and synthesize an answer.
    Optionally attempt OpenAI 'web_search' fallback if the assistant says "I don't know."
    Returns (answer_text, ctx_items).
    """
    client = ensure_client(provider)

    ctx_text, ctx_items, _, _ = retrieve_context_for_query(
        client=client,
        embed_model=embed_model,
        index=index,
        corpus_chunks=corpus_chunks,
        user_query=user_query,
        use_synonyms=use_synonyms,
        top_k=top_k,
        selected_clusters=selected_clusters,
        selected_breeds=selected_breeds,
        selected_traits=selected_traits,
        selected_genes=selected_genes,
    )

    answer = synthesize_answer(
        client, provider, chat_model, question=user_query, ctx_text=ctx_text, user_role=user_role
    )

    # Optional: if the model declares lack of knowledge, try OpenAI Responses web_search
    if allow_openai_web_fallback and _looks_like_idk(answer):
        alt = answer_with_openai_web_search(
            client=client,
            chat_model=chat_model,
            question=user_query,
            user_role=user_role,
        )
        if alt and isinstance(alt, str) and alt.strip():
            answer = alt

    return answer, ctx_items


def _looks_like_idk(text: str) -> bool:
    t = (text or "").lower()
    return any(
        p in t
        for p in [
            "i don't know",
            "i do not know",
            "not present in the context",
            "cannot find this in the context",
            "no information in the provided context",
        ]
    )