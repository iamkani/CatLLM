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

# Pre-build the shared RAG store from CLUStSTERS on cold start (best-effort)
try:
    from catllm.bootstrap_rag import bootstrap_shared_store
    bootstrap_shared_store()
except Exception as _e:
    st.warning(f"RAG bootstrap skipped: {_e}")

# Import the existing UI module so it renders as before
from catllm import ui_streamlit  # noqa: F401  (import side-effects render the UI)
