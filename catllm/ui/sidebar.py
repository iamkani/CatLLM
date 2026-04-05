import streamlit as st
from typing import List
from ..roles import normalize_role, UserRole, ROLE_GUIDANCE

def render_sidebar(state):
    st.header("Settings")

    st.markdown("---")
    st.subheader("Answer style (role)")
    # --- Role selector (explicit, no inference) ---
    role_choice = st.selectbox(
        "User role",
        ("Association Analyst","Buyer / Feeder","Genetic Advisor","Independent Rancher"),
        index=0,
        key="role_select",
    )
    user_role: UserRole = normalize_role(role_choice)

    # --- Role change handler: wipe chat + reload ---
    last_role = st.session_state.get("last_role")
    if last_role is None:
        st.session_state["last_role"] = user_role
    elif last_role != user_role:
        # clear chat/session bits
        st.session_state["messages"] = []
        st.session_state.pop("new_user_q", None)
        st.session_state["clear_chat_input"] = True  # our chat-input fix will clear on next run
        st.session_state["last_role"] = user_role
        st.session_state["user_role"] = user_role
        st.toast(f"Switched role to {user_role}. Cleared chat.")
        st.rerun()

    # Keep role available elsewhere
    st.session_state["user_role"] = user_role

    # --- Model provider & models ---
    provider = st.selectbox("Model Provider", ["OpenAI", "Azure OpenAI", "Local (OpenAI-compatible)"], key="provider")
    base_url = ""
    if provider == "OpenAI":
        chat_model  = st.text_input("OpenAI chat model", value=state.get("chat_model","gpt-4o-mini"), key="chat_model")
        embed_model = st.text_input("OpenAI embedding model", value=state.get("embed_model","text-embedding-3-large"), key="embed_model")
        st.caption("Set OPENAI_API_KEY in your environment.")
    elif provider == "Azure OpenAI":
        chat_model  = st.text_input("Azure chat deployment name", value=state.get("chat_model",""), key="chat_model")
        embed_model = st.text_input("Azure embeddings deployment name", value=state.get("embed_model",""), key="embed_model")
        st.caption("Set AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION.")
    else:  # Local
        base_url    = st.text_input("Server base URL", value=state.get("base_url","http://localhost:11434/v1"), key="base_url")
        chat_model  = st.text_input("Chat model name", value=state.get("chat_model","mistral"), key="chat_model")
        embed_model = st.text_input("Embedding model name", value=state.get("embed_model","nomic-embed-text"), key="embed_model")
        st.caption("Supports vLLM, Ollama, LM Studio, llama.cpp server, etc.")

    chunk_size   = st.slider("Chunk size", 400, 1400, state.get("chunk_size",900), 50, key="chunk_size")
    overlap      = st.slider("Chunk overlap", 0, 300, state.get("overlap",120), 10, key="overlap")
    smart_chunk  = st.checkbox("Paragraph-aware chunking", value=state.get("smart_chunk", True), key="smart_chunk")
    top_k      = st.slider("Top-K passages", 1, 10, state.get("top_k",5), 1, key="top_k")

    st.markdown("---")
    st.write("**Upload documents** (PDF, TXT, MD, CSV).")
    uploads = st.file_uploader("Add/replace corpus", type=["pdf","txt","md","csv"], accept_multiple_files=True)

    st.subheader("OCR fallback (optional)")
    use_ocr = st.checkbox("Use OCR if PDF text is empty", value=state.get("use_ocr", False), key="use_ocr")
    ocr_max = st.slider("OCR: max pages", 1, 50, state.get("ocr_max_pages",10), 1, key="ocr_max_pages")
    ocr_dpi = st.slider("OCR: DPI", 100, 400, state.get("ocr_dpi",200), 25, key="ocr_dpi")

    st.markdown("---")
    st.subheader("Bulk folder ingest")
    folder_path = st.text_input("Folder path (local)", value=state.get("folder_path",""), key="folder_path")
    recursive   = st.checkbox("Recurse into subfolders", value=state.get("recursive",True), key="recursive")
    exts        = st.multiselect("File types", ["pdf","txt","md","csv"], default=state.get("exts",["pdf","txt","md","csv"]), key="exts")
    max_files   = st.slider("Max files to index", 1, 1000, state.get("max_files",200), 1, key="max_files")
    max_size_mb = st.slider("Skip files larger than (MB)", 1, 500, state.get("max_size_mb",50), 1, key="max_size_mb")
    do_scan     = st.button("Scan folder & index", key="do_scan")

    st.markdown("---")
    st.subheader("Persistence")
    store_dir = st.text_input("Store directory", value=state.get("store_dir",".rag_store"), key="store_dir")
    c1, c2, c3 = st.columns(3)
    with c1: do_load  = st.button("Load saved", key="do_load")
    with c2: do_save  = st.button("Save now", key="do_save")
    with c3: autosave = st.checkbox("Autosave after ingest", value=state.get("autosave",True), key="autosave")

    st.markdown("---")
    st.subheader("Query options")
    use_synonyms  = st.checkbox("Expand queries with synonyms", value=state.get("use_synonyms",True), key="use_synonyms")
    show_expanded = st.checkbox("Show expanded query", value=state.get("show_expanded",False), key="show_expanded")
    use_hybrid    = st.checkbox("Hybrid search (BM25 + vector)", value=state.get("use_hybrid", False), key="use_hybrid")
    if provider == "OpenAI":
        web_fallback = st.checkbox("Web search fallback if no context found", value=state.get("web_fallback", False), key="web_fallback")
    else:
        web_fallback = False

    # --- Filters (populated from ingested corpus) ---
    corpus = state.get("corpus_chunks", [])
    if corpus:
        st.markdown("---")
        st.subheader("Filters")
        all_clusters = sorted({c.get("meta", {}).get("cluster", "") for c in corpus} - {""})
        all_breeds   = sorted({b for c in corpus for b in c.get("meta", {}).get("breeds", [])})
        all_traits   = sorted({t for c in corpus for t in c.get("meta", {}).get("traits", [])})
        all_genes    = sorted({g for c in corpus for g in c.get("meta", {}).get("genes", [])})
        if all_clusters:
            st.multiselect("Clusters", all_clusters, default=state.get("selected_clusters", []), key="selected_clusters")
        if all_breeds:
            st.multiselect("Breeds", all_breeds, default=state.get("selected_breeds", []), key="selected_breeds")
        if all_traits:
            st.multiselect("Traits", all_traits, default=state.get("selected_traits", []), key="selected_traits")
        if all_genes:
            st.multiselect("Genes", all_genes, default=state.get("selected_genes", []), key="selected_genes")

    return dict(
        user_role=user_role, provider=provider, chat_model=chat_model, embed_model=embed_model, base_url=base_url,
        chunk_size=chunk_size, overlap=overlap, top_k=top_k, smart_chunk=smart_chunk,
        uploads=uploads, use_ocr=use_ocr, ocr_max_pages=ocr_max, ocr_dpi=ocr_dpi,
        folder_path=folder_path, recursive=recursive, exts=exts, max_files=max_files, max_size_mb=max_size_mb,
        do_scan=do_scan, store_dir=store_dir, do_load=do_load, do_save=do_save, autosave=autosave,
        use_synonyms=use_synonyms, show_expanded=show_expanded, use_hybrid=use_hybrid, web_fallback=web_fallback,
    )