import os
import traceback
import inspect
from pathlib import Path

try:
    import streamlit as st
except Exception:
    class _Stub:
        def write(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def info(self, *a, **k): pass
        def error(self, *a, **k): pass
        @property
        def session_state(self):
            class _SS(dict): pass
            return _SS()
    st = _Stub()

DEFAULT_BOOTSTRAP_DIR = os.getenv('RAG_BOOTSTRAP_FOLDER', 'CLUSTERS')
DEFAULT_STORE_DIR = os.getenv('RAG_STORE_DIR', 'data/shared_store')


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _resolve_folder(folder: str) -> str:
    """Try multiple locations for a relative folder name:
    - as given
    - relative to CWD
    - relative to the project root (two levels up from this file)
    """
    p = Path(folder)
    if p.is_dir():
        return str(p)
    cwd = Path.cwd() / folder
    if cwd.is_dir():
        return str(cwd)
    project_root = Path(__file__).resolve().parent.parent / folder
    if project_root.is_dir():
        return str(project_root)
    return str(p)


def _extract_store(retval):
    """Try multiple ways to obtain a store object after build.
    Returns (store_obj or None).
    """
    if isinstance(retval, tuple) and len(retval) == 2:
        _idx, _store = retval
        return _store
    if hasattr(st, 'session_state'):
        for key in ('store','pipeline_store','rag_store','vector_store'):
            try:
                v = st.session_state.get(key)
            except Exception:
                v = None
            if v is not None:
                return v
    return None


def _get_models():
    chat = os.getenv('OPENAI_CHAT_MODEL') or 'gpt-4o-mini'
    embed = os.getenv('OPENAI_EMBEDDINGS_MODEL') or 'text-embedding-3-large'
    return chat, embed


def bootstrap_shared_store():
    """Build a shared RAG store from the CLUSTERS folder on cold deploy.

    Controlled by env vars:
      - RAG_BOOTSTRAP_FOLDER (default: 'CLUSTERS')
      - RAG_STORE_DIR (default: 'data/shared_store')
    """
    folder = _resolve_folder(DEFAULT_BOOTSTRAP_DIR)
    store_dir = DEFAULT_STORE_DIR

    if not os.path.isdir(folder):
        st.info(f"Bootstrap folder not found: {folder} — skipping prebuild.")
        return False

    _ensure_dir(store_dir)

    try:
        from . import pipeline as pl
    except Exception as e:
        st.warning(f"Could not import pipeline: {e}")
        return False

    ingest_folder_to_chunks = getattr(pl, 'ingest_folder_to_chunks', None)
    build_index_from_chunks = getattr(pl, 'build_index_from_chunks', None)
    if not ingest_folder_to_chunks or not build_index_from_chunks:
        st.warning("Pipeline functions missing (ingest_folder_to_chunks/build_index_from_chunks) — skipping bootstrap.")
        return False

    try:
        chunks = ingest_folder_to_chunks(folder)
        # Always make chunks available via session_state for zero-arg pipelines
        try:
            st.session_state['chunks'] = chunks
        except Exception:
            pass

        chat_model, embed_model = _get_models()
        # Attempt 1: newer signature with keyword-only args
        try:
            retval = build_index_from_chunks(
                provider='openai',
                chat_model=chat_model,
                embed_model=embed_model,
                corpus_chunks=chunks,
            )
        except TypeError:
            # Try capitalized provider
            try:
                retval = build_index_from_chunks(
                    provider='OpenAI',
                    chat_model=chat_model,
                    embed_model=embed_model,
                    corpus_chunks=chunks,
                )
            except TypeError:
                # Fallback: zero-arg, then positional with chunks
                try:
                    retval = build_index_from_chunks()
                except TypeError:
                    retval = build_index_from_chunks(chunks)

        store = _extract_store(retval)
        if store is None:
            st.warning("Built index but could not locate store object to persist.")
            return False

        try:
            from .persistence import save_store
            save_store(store, store_dir)
            st.info(f"Shared store built and saved to {store_dir}.")
            return True
        except Exception as e2:
            st.warning(f"Built index, but could not persist shared store: {e2}")
            return False
    except Exception:
        st.error("Bootstrap ingest failed. See logs.")
        st.write(traceback.format_exc())
        return False
