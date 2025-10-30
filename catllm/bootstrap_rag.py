import os
import traceback
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


def _normalize_chunks(raw):
    """Return a list[dict] shaped as {"text": str, ...}.
    Accepts: list[dict], list[str], tuple(list, fails), {"chunks": list}, etc.
    """
    # Unwrap tuple: (chunks, failures)
    if isinstance(raw, tuple) and len(raw) >= 1:
        raw = raw[0]

    # Unwrap mapping with key 'chunks'
    if isinstance(raw, dict) and 'chunks' in raw:
        raw = raw['chunks']

    # Now ensure it's a list
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        # Single string
        if isinstance(raw, str):
            return [{"text": raw}]
        # Unknown shape
        return []

    normalized = []
    for item in raw:
        if item is None:
            continue
        if isinstance(item, str):
            t = item
            if t.strip():
                normalized.append({"text": t})
            continue
        if isinstance(item, dict):
            # accept common keys
            if 'text' in item and isinstance(item['text'], str):
                normalized.append(item)
                continue
            if 'content' in item and isinstance(item['content'], str):
                d = dict(item)
                d.setdefault('text', d.pop('content'))
                normalized.append(d)
                continue
            if 'page_content' in item and isinstance(item['page_content'], str):
                d = dict(item)
                d.setdefault('text', d.pop('page_content'))
                normalized.append(d)
                continue
            # Last resort: stringify whatever is there
            s = str(item)
            if s.strip():
                normalized.append({"text": s})
            continue
        # Lists or other types: stringify
        s = str(item)
        if s.strip():
            normalized.append({"text": s})

    return normalized


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

    # Support both new and legacy ingest entrypoints
    ingest_fn = getattr(pl, 'ingest_folder_to_chunks', None) or getattr(pl, 'ingest_folder', None)
    build_index_from_chunks = getattr(pl, 'build_index_from_chunks', None)
    if not ingest_fn or not build_index_from_chunks:
        st.warning("Pipeline functions missing — skipping bootstrap.")
        return False

    try:
        raw = ingest_fn(folder)
        chunks = _normalize_chunks(raw)
        if not chunks:
            st.warning("No chunks produced from folder; skipping.")
            return False

        try:
            st.session_state['chunks'] = chunks
        except Exception:
            pass

        chat_model, embed_model = _get_models()

        # Preferred: keyword-only signature (provider/chat_model/embed_model/corpus_chunks)
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
                # Fallbacks for very old signatures
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
