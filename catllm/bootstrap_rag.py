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

# --- Helpers ---------------------------------------------------------------

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


# --- Chunk normalization & safety split -----------------------------------

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


def _approx_tokens(text: str, chars_per_token: int = 4) -> int:
    try:
        n = len(text)
    except Exception:
        n = len(str(text))
    return max(1, n // max(1, chars_per_token))


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text on friendly boundaries (\n\n, then sentences) falling back to hard slices.
    """
    if len(text) <= max_chars:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        # try to break at last paragraph boundary within window
        last_para = chunk.rfind("\n\n")
        if last_para >= max_chars * 2 // 3:
            end = start + last_para
            chunk = text[start:end]
        else:
            # try sentence boundary
            last_sent = chunk.rfind('. ')
            if last_sent >= max_chars * 2 // 3:
                end = start + last_sent + 1
                chunk = text[start:end]
        out.append(chunk)
        start = end
    return [s for s in (c.strip() for c in out) if s]


def _enforce_token_limit(chunks: list[dict], max_tokens: int = 7500, chars_per_token: int = 4) -> list[dict]:
    max_chars = max_tokens * chars_per_token
    safe = []
    for i, ch in enumerate(chunks):
        t = ch.get('text', '') if isinstance(ch, dict) else str(ch)
        if not isinstance(t, str):
            t = str(t)
        if _approx_tokens(t, chars_per_token) <= max_tokens:
            safe.append(ch if isinstance(ch, dict) else { 'text': t })
            continue
        # split
        parts = _split_text(t, max_chars)
        for j, part in enumerate(parts):
            if not part:
                continue
            meta = dict(ch) if isinstance(ch, dict) else {}
            meta['text'] = part
            meta['__split__'] = True
            meta['__parent_index__'] = i
            meta['__part__'] = j
            safe.append(meta)
    return safe


# --- Main -----------------------------------------------------------------

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

        # Safety split to avoid OpenAI 8k token limit on embeddings
        chunks = _enforce_token_limit(chunks, max_tokens=7500, chars_per_token=4)

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
