# catllm/ui/tabchat.py
from __future__ import annotations
import streamlit as st

from ..llm import ensure_client, synthesize_answer
from ..embeddings import embed_texts
from ..indexer import top_k_search
from ..retrieval import format_context
from ..prompts import build_system_prompt_for_role
from ..roles import apply_role_style
from ..utils_text import expand_query


def _get(key: str, default=None):
    return st.session_state.get(key, default)


def _set(key: str, value):
    st.session_state[key] = value


def render_tab_chat(state):
    """
    Classic chat UX with input anchored at the bottom:
    - We render all chat content first.
    - We place st.chat_input() as the LAST widget.
    - On submit, stash the text and st.rerun(); the next run renders the new Q/A above,
      then shows the input at the very bottom again.
    """
    # --- Read sidebar-configured options (do not render any widgets here) ---
    provider      = _get("provider", "OpenAI")
    chat_model    = _get("chat_model", "gpt-4o-mini")
    embed_model   = _get("embed_model", "text-embedding-3-large")
    top_k         = _get("top_k", 5)
    use_synonyms  = _get("use_synonyms", True)
    show_expanded = _get("show_expanded", False)
    user_role     = _get("user_role", "Independent Rancher")

    # --- Guard: need an index to chat ---
    if _get("index") is None or len(_get("corpus_chunks", [])) == 0:
        st.info("Upload documents, scan a folder, or load a saved store to start chatting.")
        # Still render the input at the very bottom for consistency
        _chat_input_bottom(user_role)
        return

    # --- If the previous run stored a new user question, process it now ---
    pending_q = st.session_state.pop("new_user_q", None)
    if pending_q:
        # Show user question in history immediately
        st.session_state.setdefault("messages", []).append({"role": "user", "content": pending_q})

        # Retrieval + answer generation
        client = ensure_client(provider)

        expanded_q = expand_query(pending_q) if use_synonyms else pending_q
        if show_expanded and expanded_q != pending_q:
            # Quick caption just above the answer for transparency
            st.caption(f"Expanded query: {expanded_q}")

        q_vec = embed_texts(client, [expanded_q], embed_model)

        search_k = top_k
        if _get("selected_clusters"):
            search_k = max(50, top_k * 5)

        I, _D = top_k_search(_get("index"), q_vec, k=search_k)
        candidates = [_get("corpus_chunks")[i] for i in I]

        # Optional filtering by cluster/tags
        selected_clusters = _get("selected_clusters", [])
        selected_breeds   = _get("selected_breeds", [])
        selected_traits   = _get("selected_traits", [])
        selected_genes    = _get("selected_genes", [])

        if selected_clusters:
            filt = [c for c in candidates if c.get("meta", {}).get("cluster", "") in selected_clusters]
            if len(filt) >= top_k:
                candidates = filt

        def _match_tags(meta):
            if selected_breeds and not (set(meta.get("breeds", [])) & set(selected_breeds)):
                return False
            if selected_traits and not (set(meta.get("traits", [])) & set(selected_traits)):
                return False
            if selected_genes and not (set(meta.get("genes", [])) & set(selected_genes)):
                return False
            return True

        if selected_breeds or selected_traits or selected_genes:
            filt = [c for c in candidates if _match_tags(c.get("meta", {}))]
            if len(filt) >= top_k:
                candidates = filt

        ctx_items = candidates[:top_k]
        ctx_text  = format_context(ctx_items)

        # Role-aware system prompt (explicit; we never infer)
        system_prompt = build_system_prompt_for_role(user_role)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"QUESTION:\n{pending_q}\n\nCONTEXT:\n{ctx_text}"},
        ]

        with st.spinner("Thinking…"):
            raw = synthesize_answer(client, provider, chat_model, messages)

        answer = apply_role_style(user_role, raw)

        # Append assistant response to history
        st.session_state["messages"].append({"role": "assistant", "content": answer})

        # Also stash last retrieval context for the expander (optional)
        st.session_state["last_ctx_items"] = ctx_items

    # --- Render the FULL chat history (user & assistant) ABOVE the input ---
    for m in _get("messages", []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Optional transparency block for last turn
    if "last_ctx_items" in st.session_state and st.session_state["last_ctx_items"]:
        ctx_items = st.session_state["last_ctx_items"]
        with st.expander("Retrieved context"):
            for idx, item in enumerate(ctx_items, 1):
                meta = item.get("meta", {})
                st.markdown(f"**[{idx}] {meta.get('title','Unknown')}**")
                st.caption(
                    f"Cluster: {meta.get('cluster','')}  |  "
                    f"Breeds: {', '.join(meta.get('breeds', []))}  |  "
                    f"Traits: {', '.join(meta.get('traits', []))}  |  "
                    f"Genes: {', '.join(meta.get('genes', []))}  |  "
                    f"Source: {meta.get('source','')}"
                )
                txt = item.get("text", "")
                st.write(txt[:1200] + ("…" if len(txt) > 1200 else ""))

    # --- Finally, render the INPUT at the very bottom ---
    _chat_input_bottom(user_role)


def _chat_input_bottom(user_role: str):
    """Place chat input at the bottom; on submit, stash & rerun.
    We clear the widget value on the NEXT run (before rendering it) to avoid
    Streamlit's 'cannot modify after instantiation' error.
    """
    # If the previous run set a clear flag, drop the old widget state now (before creating it)
    if st.session_state.pop("clear_chat_input", False):
        st.session_state.pop("chat_input_bottom", None)

    placeholder = f"Ask about cattle genetics (role: {user_role})…"
    new_text = st.chat_input(placeholder, key="chat_input_bottom")

    if new_text:
        # Stash and trigger a rerun. On the next run, we'll clear the widget state first.
        st.session_state["new_user_q"] = new_text
        st.session_state["clear_chat_input"] = True
        st.rerun()