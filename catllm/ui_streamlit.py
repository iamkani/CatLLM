# catllm/ui_streamlit.py
from __future__ import annotations

import os
import json
from typing import Dict, List, Sequence

import numpy as np
import streamlit as st

from .pipeline import (
    ingest_folder_to_chunks,
    ingest_uploads_to_chunks,
    build_index_from_chunks,
    save_pipeline_store,
    load_pipeline_store,
    generate_answer_from_rag,
)
from .tagging import list_all_tags_from_chunks, ROLE_LIST


st.set_page_config(page_title="CattLLM — Mini RAG Chat", page_icon="🐮", layout="wide")
st.title("🐮 CattLLM — Mini RAG Chat")


# -----------------------------
# Sidebar: settings
# -----------------------------
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox("Model Provider", ["OpenAI", "Azure OpenAI"])
    if provider == "OpenAI":
        chat_model = st.text_input("OpenAI chat model", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
        embed_model = st.text_input("OpenAI embedding model", os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large"))
        st.caption("Set OPENAI_API_KEY in your environment.")
    else:
        chat_model = st.text_input("Azure chat deployment name", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", ""))
        embed_model = st.text_input("Azure embeddings deployment name", os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", ""))
        st.caption("Set AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION in your environment.")

    st.markdown("---")
    st.subheader("Chunking & Retrieval")
    chunk_size = st.slider("Chunk size (tokens-ish)", 400, 1400, 900, 50)
    overlap = st.slider("Chunk overlap", 0, 300, 120, 10)
    top_k = st.slider("Top-K passages", 1, 10, 5, 1)
    use_synonyms = st.checkbox("Expand queries with synonyms", value=True)
    allow_openai_web_fallback = st.checkbox("OpenAI web-search fallback if RAG says 'I don't know'", value=False)

    st.markdown("---")
    st.subheader("Persistence")
    store_dir = st.text_input("Store directory", value=".rag_store")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        do_load = st.button("Load saved")
    with col_b:
        do_save = st.button("Save now")
    with col_c:
        autosave = st.checkbox("Autosave after ingest", value=True)

    st.markdown("---")
    st.subheader("Bulk folder ingest")
    folder_path = st.text_input("Folder path (local)", value="")
    recursive = st.checkbox("Recurse into subfolders", value=True)
    exts = st.multiselect("File types", ["pdf", "txt", "md", "csv"], default=["pdf", "txt", "md", "csv"])
    max_files = st.slider("Max files to index", 1, 1000, 200, 1)
    max_size_mb = st.slider("Skip files larger than (MB)", 1, 500, 50, 1)
    do_scan = st.button("Scan folder & index")

    st.markdown("---")
    st.subheader("Manual uploads")
    uploads = st.file_uploader("Upload PDF/TXT/MD/CSV", type=["pdf", "txt", "md", "csv"], accept_multiple_files=True)
    use_ocr = st.checkbox("Use OCR if PDF text is empty", value=False)
    ocr_max_pages = st.slider("OCR: max pages", 1, 50, 10, 1)
    ocr_dpi = st.slider("OCR: DPI", 100, 400, 200, 25)

    st.markdown("---")
    st.subheader("Role & Filters")
    user_role = st.selectbox("User Role (answer style)", ROLE_LIST, index=ROLE_LIST.index("Independent Rancher"))

    # dynamic cluster/tag filters will be populated after index is built


# -----------------------------
# Session state
# -----------------------------
if "corpus_chunks" not in st.session_state:
    st.session_state.corpus_chunks: List[Dict] = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict] = []
if "fails" not in st.session_state:
    st.session_state.fails: List[str] = []


# -----------------------------
# Load store
# -----------------------------
if do_load:
    cc, embs, idx = load_pipeline_store(store_dir)
    if cc is None:
        st.warning(f"No saved store found in {store_dir}")
    else:
        st.session_state.corpus_chunks = cc
        st.session_state.embeddings = embs
        st.session_state.index = idx
        st.success(f"Loaded {len(cc)} chunks from {store_dir}")


# -----------------------------
# Manual uploads
# -----------------------------
if uploads:
    chunks = ingest_uploads_to_chunks(
        uploads,
        use_ocr=use_ocr,
        ocr_max_pages=ocr_max_pages,
        ocr_dpi=ocr_dpi,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    st.session_state.corpus_chunks = chunks

    with st.spinner("Embedding & indexing…"):
        embs, idx = build_index_from_chunks(
            provider=provider,
            chat_model=chat_model,
            embed_model=embed_model,
            corpus_chunks=st.session_state.corpus_chunks,
        )
        st.session_state.embeddings = embs
        st.session_state.index = idx

    st.success(f"Ingested {len(chunks)} chunks from {len(uploads)} upload(s).")
    if autosave:
        save_pipeline_store(store_dir, st.session_state.corpus_chunks, st.session_state.embeddings, st.session_state.index)
        st.caption(f"Autosaved index to {store_dir}")


# -----------------------------
# Bulk ingest
# -----------------------------
if do_scan and folder_path.strip():
    status = st.status("Scanning folder…", state="running")
    scan_bar = status.progress(0)
    current_file = status.empty()

    def _cb(i, total, path, phase):
        pct = int(i / max(1, total) * 100)
        scan_bar.progress(pct)
        phase_title = {"scan": "Reading", "parse": "Parsing", "done": "Finished"}.get(phase, phase)
        current_file.markdown(f"**{i}/{total}** — **{phase_title}**: `{path}`")

    chunks, fails = ingest_folder_to_chunks(
        folder_path=folder_path.strip(),
        recursive=recursive,
        exts=exts,
        max_files=max_files,
        max_size_mb=max_size_mb,
        use_ocr=use_ocr,
        ocr_max_pages=ocr_max_pages,
        ocr_dpi=ocr_dpi,
        chunk_size=chunk_size,
        overlap=overlap,
        progress_cb=_cb,
    )
    st.session_state.fails = fails
    if not chunks:
        status.update(label="No files found (check path, extensions, size filter).", state="error")
    else:
        st.session_state.corpus_chunks = chunks
        status.update(label=f"Embedding {len(chunks)} chunks…", state="running")

        with st.spinner("Embedding & indexing…"):
            embs, idx = build_index_from_chunks(
                provider=provider,
                chat_model=chat_model,
                embed_model=embed_model,
                corpus_chunks=st.session_state.corpus_chunks,
            )
            st.session_state.embeddings = embs
            st.session_state.index = idx

        status.update(label=f"Ingested {len(chunks)} chunks from folder.", state="complete")
        if fails:
            with st.expander(f"{len(fails)} file(s) failed to parse (click to view)"):
                for f in fails:
                    st.write(f)
        if autosave:
            save_pipeline_store(store_dir, st.session_state.corpus_chunks, st.session_state.embeddings, st.session_state.index)
            st.caption(f"Autosaved index to {store_dir}")


# -----------------------------
# Role & Filters (dynamic)
# -----------------------------
clusters, breeds, traits, genes = list_all_tags_from_chunks(st.session_state.corpus_chunks)
with st.sidebar:
    selected_clusters = st.multiselect("Clusters to include", clusters, default=[])
    selected_breeds = st.multiselect("Breeds", breeds, default=[])
    selected_traits = st.multiselect("Traits", traits, default=[])
    selected_genes = st.multiselect("Genes", genes, default=[])


# -----------------------------
# Tabs
# -----------------------------
tab_chat, tab_generate = st.tabs(["🗨️ Chat", "🧪 Q&A Generator"])


# -----------------------------
# Tab: Chat
# -----------------------------
with tab_chat:
    if st.session_state.index is None or len(st.session_state.corpus_chunks) == 0:
        st.info("Load/build an index first (Upload / Bulk ingest / Load saved).")
    else:
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask about cattle genetics, EBVs/EPDs, SNP effects, etc.")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.spinner("Thinking…"):
                answer, ctx_items = generate_answer_from_rag(
                    provider=provider,
                    chat_model=chat_model,
                    embed_model=embed_model,
                    index=st.session_state.index,
                    corpus_chunks=st.session_state.corpus_chunks,
                    user_query=user_q,
                    user_role=user_role,
                    use_synonyms=use_synonyms,
                    top_k=top_k,
                    selected_clusters=selected_clusters,
                    selected_breeds=selected_breeds,
                    selected_traits=selected_traits,
                    selected_genes=selected_genes,
                    allow_openai_web_fallback=allow_openai_web_fallback,
                )

            with st.chat_message("assistant"):
                st.markdown(answer)

            with st.expander("Retrieved context"):
                for i, item in enumerate(ctx_items, 1):
                    meta = item.get("meta", {})
                    st.markdown(f"**[{i}] {meta.get('title','Unknown')}**")
                    st.caption(
                        f"Cluster: {meta.get('cluster','')}  |  "
                        f"Breeds: {', '.join(meta.get('breeds', []))}  |  "
                        f"Traits: {', '.join(meta.get('traits', []))}  |  "
                        f"Genes: {', '.join(meta.get('genes', []))}  |  "
                        f"Source: {meta.get('path','') or meta.get('source','')}"
                    )
                    st.write(item["text"][:1200] + ("…" if len(item["text"]) > 1200 else ""))

            st.session_state.chat_history.append({"role": "assistant", "content": answer})


# -----------------------------
# Tab: Q&A Generator
# -----------------------------
with tab_generate:
    st.subheader("Generate Q&A from CSV/XLSX using the current RAG index")
    if st.session_state.index is None or len(st.session_state.corpus_chunks) == 0:
        st.info("Load/build an index first (Upload / Bulk ingest / Load saved).")
    else:
        qa_file = st.file_uploader("Upload CSV or XLSX with questions (and optional answers)", type=["csv", "xlsx", "xls"])
        infer_topk = st.slider("Top-K passages per question", 1, 10, 5, 1)
        run_btn = st.button("Generate Q&A")

        if run_btn and qa_file is not None:
            import pandas as pd

            # read CSV/XLSX
            if qa_file.name.lower().endswith((".xlsx", ".xls")):
                try:
                    import openpyxl  # noqa: F401
                except Exception:
                    st.error("Please install openpyxl to read .xlsx/.xls: pip install openpyxl")
                    st.stop()
                df = pd.read_excel(qa_file)
            else:
                df = pd.read_csv(qa_file)

            # detect columns
            q_candidates = ["question", "q", "prompt", "query"]
            a_candidates = ["answer", "a", "response"]
            q_col = next((c for c in df.columns if c.lower() in q_candidates), None)
            a_col = next((c for c in df.columns if c.lower() in a_candidates), None)
            if not q_col:
                st.error("Could not find a question column (question/q/prompt/query).")
                st.stop()

            results: List[Dict] = []
            progress = st.progress(0)
            total = len(df)

            # small embed (re-using the retrieval code path by calling generate_answer_from_rag)
            for i, row in df.iterrows():
                qtext = str(row[q_col]).strip()
                if not qtext:
                    continue

                ans, ctx_items = generate_answer_from_rag(
                    provider=provider,
                    chat_model=chat_model,
                    embed_model=embed_model,
                    index=st.session_state.index,
                    corpus_chunks=st.session_state.corpus_chunks,
                    user_query=qtext,
                    user_role=user_role,
                    use_synonyms=use_synonyms,
                    top_k=infer_topk,
                    selected_clusters=selected_clusters,
                    selected_breeds=selected_breeds,
                    selected_traits=selected_traits,
                    selected_genes=selected_genes,
                    allow_openai_web_fallback=allow_openai_web_fallback,
                )

                titles = [c.get("meta", {}).get("title", "Unknown") for c in ctx_items]
                paths = [c.get("meta", {}).get("path", "") for c in ctx_items]
                clusters = [c.get("meta", {}).get("cluster", "") for c in ctx_items]
                breeds = sorted({b for c in ctx_items for b in c.get("meta", {}).get("breeds", [])})
                traits = sorted({t for c in ctx_items for t in c.get("meta", {}).get("traits", [])})
                genes = sorted({g for c in ctx_items for g in c.get("meta", {}).get("genes", [])})

                results.append(
                    {
                        "question": qtext,
                        "answer": ans,
                        "citations": "; ".join(titles),
                        "doc_paths": "; ".join(paths),
                        "clusters": "; ".join([x for x in clusters if x]),
                        "breeds": "; ".join(breeds),
                        "traits": "; ".join(traits),
                        "genes": "; ".join(genes),
                    }
                )

                if (i + 1) % 5 == 0 or (i + 1) == total:
                    progress.progress(int((i + 1) / max(1, total) * 100))

            out_df = pd.DataFrame(results)
            st.success(f"Generated {len(results)} Q&A rows.")
            st.dataframe(out_df, use_container_width=True)

            st.download_button(
                "Download CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="generated_qa.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download JSONL",
                data="\n".join([json.dumps(r, ensure_ascii=False) for r in results]).encode("utf-8"),
                file_name="generated_qa.jsonl",
                mime="application/json",
            )

# -----------------------------
# Save now
# -----------------------------
if do_save and st.session_state.index is not None and st.session_state.corpus_chunks:
    save_pipeline_store(store_dir, st.session_state.corpus_chunks, st.session_state.embeddings, st.session_state.index)
    st.success(f"Saved store to {store_dir}")