# Keep package init lightweight, but add compatibility shims.
# Do NOT import heavy submodules here except minimal patches that unblock legacy imports.

__all__ = []

# --- Persistence re-export for legacy imports --------------------------------
try:
    from . import utils_text as _ut
    from .persistence import save_store as _save_store, load_store as _load_store
    setattr(_ut, 'save_store', _save_store)
    setattr(_ut, 'load_store', _load_store)
except Exception:
    pass

# --- Tagging API compatibility ------------------------------------------------
# Some versions of the app expect `list_all_tags_from_chunks` and `ROLE_LIST` in catllm.tagging.
try:
    from . import tagging as _tg
    # ROLE_LIST: try to pull from roles.py first; else define a sane default
    try:
        from .roles import ROLE_LIST as _ROLE_LIST
    except Exception:
        _ROLE_LIST = ["Root", "Admin", "User"]
    if not hasattr(_tg, 'ROLE_LIST'):
        setattr(_tg, 'ROLE_LIST', _ROLE_LIST)
    # list_all_tags_from_chunks: collect union of known tag fields across chunks
    if not hasattr(_tg, 'list_all_tags_from_chunks'):
        def _list_all_tags_from_chunks(chunks):
            tags = set()
            if not chunks:
                return []
            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                # common tag-like keys used in this project
                for key in ('tags','tag','traits','genes','clusters','categories','labels'):
                    val = ch.get(key)
                    if not val:
                        continue
                    if isinstance(val, (list, tuple, set)):
                        for t in val:
                            if isinstance(t, str) and t.strip():
                                tags.add(t.strip())
                    elif isinstance(val, str):
                        if ',' in val:
                            for t in val.split(','):
                                if t.strip():
                                    tags.add(t.strip())
                        elif val.strip():
                            tags.add(val.strip())
            return sorted(tags)
        setattr(_tg, 'list_all_tags_from_chunks', _list_all_tags_from_chunks)
except Exception:
    pass
