import os
import traceback

try:
    import streamlit as st
except Exception:
    class _Stub:
        def write(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def info(self, *a, **k): pass
        def error(self, *a, **k): pass
    st = _Stub()

DEFAULT_BOOTSTRAP_DIR = os.getenv('RAG_BOOTSTRAP_FOLDER', 'CLUStSTERS')
DEFAULT_STORE_DIR = os.getenv('RAG_STORE_DIR', 'data/shared_store')


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def bootstrap_shared_store():
    """On cold deploy, build a shared RAG store from the bootstrap folder.

    Controlled by env vars:
      - RAG_BOOTSTRAP_FOLDER (default: 'CLUStSTERS')
      - RAG_STORE_DIR (default: 'data/shared_store')
    """
    folder = DEFAULT_BOOTSTRAP_DIR
    store_dir = DEFAULT_STORE_DIR

    if not os.path.isdir(folder):
        st.info(f"Bootstrap folder not found: {folder} — skipping prebuild.")
        return False

    _ensure_dir(store_dir)

    # Attempt to use pipeline helpers if available
    try:
        from . import pipeline as pl  # type: ignore
    except Exception as e:
        st.warning(f"Could not import pipeline: {e}")
        return False

    try:
        # Expect these symbols in pipeline; guard with getattr for robustness
        ingest_folder_to_chunks = getattr(pl, 'ingest_folder_to_chunks', None)
        build_index_from_chunks = getattr(pl, 'build_index_from_chunks', None)
    except Exception:
        ingest_folder_to_chunks = None
        build_index_from_chunks = None

    if not ingest_folder_to_chunks or not build_index_from_chunks:
        st.warning("Pipeline functions missing (ingest_folder_to_chunks/build_index_from_chunks) — skipping bootstrap.")
        return False

    try:
        chunks = ingest_folder_to_chunks(folder)
        index, store = build_index_from_chunks(chunks)
        # save via persistence if available
        try:
            from .persistence import save_store  # type: ignore
            save_store(store, store_dir)
            st.info(f"Shared store built and saved to {store_dir}.")
            return True
        except Exception as e2:
            st.warning(f"Built index, but could not persist shared store: {e2}")
            return False
    except Exception as e:
        st.error("Bootstrap ingest failed. See logs.")
        st.write(traceback.format_exc())
        return False
