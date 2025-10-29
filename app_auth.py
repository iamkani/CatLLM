import os
import streamlit as st

# Export any Streamlit Cloud secrets into environment variables
for k, v in dict(getattr(st, 'secrets', {})).items():
    os.environ.setdefault(str(k), str(v))

# Gate the app before anything else renders
from catllm.ui_auth_gate import require_login, user_bar, admin_console
current_user = require_login()
user_bar()
admin_console()

# --- Runtime compatibility shim for legacy imports ---
# Ensure pipeline's `from .utils_text import save_store, load_store` works
try:
    import catllm.utils_text as _ut
    from catllm.persistence import save_store as _save_store, load_store as _load_store
    setattr(_ut, 'save_store', _save_store)
    setattr(_ut, 'load_store', _load_store)
except Exception:
    pass

# Import the existing UI module so it renders as before
from catllm import ui_streamlit  # noqa: F401  (import side-effects render the UI)
