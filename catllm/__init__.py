# Keep package init lightweight, but add a tiny compatibility shim.
# Do NOT import heavy submodules here except for a minimal patch.

__all__ = []

# --- Compatibility shim -----------------------------------------------------
# Some legacy code imports `save_store`/`load_store` from utils_text,
# but these functions live in `catllm.persistence`.
# We re-export them onto the utils_text module so existing imports keep working.
try:
    from . import utils_text as _ut  # light import
    from .persistence import save_store as _save_store, load_store as _load_store
    setattr(_ut, 'save_store', _save_store)
    setattr(_ut, 'load_store', _load_store)
except Exception:
    # If anything fails here, the app can still boot; UI will surface if missing later.
    pass
