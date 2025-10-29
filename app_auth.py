import os
import streamlit as st

# Export any Streamlit Cloud secrets into environment variables
for k, v in dict(getattr(st, 'secrets', {})).items():
    os.environ.setdefault(str(k), str(v))

from catllm.ui_auth_gate import require_login, user_bar, admin_console
current_user = require_login()
user_bar()
admin_console()

from catllm import ui_streamlit  # noqa: F401
