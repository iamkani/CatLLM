import streamlit as st

# Gate the app before anything else renders
from catllm.ui_auth_gate import require_login, user_bar, admin_console
current_user = require_login()
user_bar()
admin_console()

# Import the existing UI module so it renders as before
# If your app previously executed logic in app.py, move it to catllm/ui_streamlit.py
from catllm import ui_streamlit  # noqa: F401  (import side-effects render the UI)
