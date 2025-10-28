# app.py
import streamlit as st
from catllm.ui.sidebar import render_sidebar
from catllm.logging_setup import setup_logging
from catllm.ingest import ingest_uploads, ingest_folder
from catllm.persistence import save_store, load_store
from catllm.embeddings import embed_texts
from catllm.indexer import build_faiss_index as build_index
from catllm.utils_text import chunk_text
from catllm.tagging import tag_text_for_meta
from catllm.llm import ensure_client
from catllm.ui.tabchat import render_tab_chat
from catllm.ui.tab_qa_generator import render_generator

logger = setup_logging()
st.set_page_config(page_title="CattLLM (RAG for Cattle Genetics)", page_icon="🐮")
st.title("🐮 CattLLM — Role-aware RAG for Cattle Genetics")
tab_chat, tab_generate = st.tabs(["🗨️ Chat","🧪 Q&A Generator"])

# Session state
st.session_state.setdefault("corpus_chunks", [])
st.session_state.setdefault("embeddings", None)
st.session_state.setdefault("index", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("user_role", "Association Analyst")
st.session_state.setdefault("system_prompt",
    "You are a cattle-genetics RAG assistant. Use ONLY provided context; if unknown, say so. Cite [#].")

with st.sidebar:
    ui = render_sidebar(st.session_state)
    provider    = ui["provider"]
    chat_model  = ui["chat_model"]
    embed_model = ui["embed_model"]

# Load store
if ui["do_load"]:
    cc, embs, idx = load_store(ui["store_dir"])
    if cc is None: st.warning(f"No saved store found in {ui['store_dir']}")
    else:
        st.session_state["corpus_chunks"] = cc
        st.session_state["embeddings"] = embs
        st.session_state["index"] = idx
        st.success(f"Loaded {len(cc)} chunks.")

# Uploads
if ui["uploads"]:
    docs = ingest_uploads(ui["uploads"], use_ocr=ui["use_ocr"], ocr_max_pages=ui["ocr_max_pages"],
                          ocr_dpi=ui["ocr_dpi"], seen_hashes=set())
    chunks=[]
    for title, text in docs:
        for ch in chunk_text(text, ui["chunk_size"], ui["overlap"]):
            meta = {"title": title, "source": title}
            meta.update(tag_text_for_meta(ch))
            chunks.append({"text": ch, "meta": meta})
    st.session_state["corpus_chunks"] = chunks

    with st.spinner("Embedding & indexing…"):
        client = ensure_client(ui["provider"])
        print("chunks_len:", len(chunks))
        if chunks:
            print("first_chunk_keys:", list(chunks[0].keys()))
            nonempty = sum(1 for c in chunks if isinstance(c.get("text"), str) and c["text"].strip())
            print("nonempty_text_chunk_count:", nonempty)
        texts = [c.get("text","").strip() for c in chunks if isinstance(c.get("text"), str) and c.get("text","").strip()]
        if not texts:
            import streamlit as st
            st.error("No extractable text found. Check your ingest folder/filters or enable OCR for scanned PDFs.")
            st.stop()

        # call with the new signature (texts before model)
        embs = embed_texts(client, texts, ui["embed_model"])
        idx  = build_index(embs)
        st.session_state["embeddings"] = embs
        st.session_state["index"] = idx
    st.success(f"Ingested {len(chunks)} chunks from {len(docs)} doc(s).")
    if ui["autosave"]:
        save_store(ui["store_dir"], chunks, embs, idx)
        st.caption(f"Autosaved to {ui['store_dir']}")

# Folder ingest
if ui["do_scan"] and ui["folder_path"].strip():
    status = st.status("Scanning folder…", state="running")
    bar = status.progress(0)
    cur = status.empty()
    def _cb(i,total,path,phase):
        bar.progress(int(i/max(1,total)*100))
        cur.markdown(f"**{i}/{total}** — `{phase}`: `{path}`")
    docs, fails = ingest_folder(ui["folder_path"].strip(), _cb, exts=ui["exts"], recursive=ui["recursive"],
                                max_files=ui["max_files"], max_size_mb=ui["max_size_mb"],
                                use_ocr=ui["use_ocr"], ocr_max_pages=ui["ocr_max_pages"], ocr_dpi=ui["ocr_dpi"],
                                seen_hashes=set())
    if not docs:
        status.update(label="No files found.", state="warning")
    else:
        chunks=[]
        for item in docs:
            title, text, extra = item if len(item)==3 else (item[0], item[1], {})
            for ch in chunk_text(text, ui["chunk_size"], ui["overlap"]):
                meta = {"title": title, "source": title}
                meta.update(extra); meta.update(tag_text_for_meta(ch))
                chunks.append({"text": ch, "meta": meta})
        st.session_state["corpus_chunks"] = chunks

        status.update(label=f"Embedding {len(chunks)} chunks…", state="running")
        client = ensure_client(ui["provider"])
        print("chunks_len:", len(chunks))
        print("first_chunk_keys:", list(chunks[0].keys()) if chunks else "no chunks")
        nonempty = [c for c in chunks if isinstance(c.get("text"), str) and c.get("text").strip()]
        print("nonempty_text_chunks:", len(nonempty))
        # build the list of usable strings before calling
        texts = [c.get("text","").strip() for c in chunks if isinstance(c.get("text"), str) and c.get("text","").strip()]
        if not texts:
            import streamlit as st
            st.error("No extractable text found. Check your ingest folder/filters or enable OCR for scanned PDFs.")
            st.stop()

        # call with the new signature (texts before model)
        embs = embed_texts(client, texts, ui["embed_model"])
        idx  = build_index(embs)
        st.session_state["embeddings"] = embs
        st.session_state["index"] = idx
        status.update(label=f"Ingested {len(chunks)} chunks from {len(docs)} file(s).", state="complete")
        if fails:
            with st.expander(f"{len(fails)} file(s) failed to parse"):
                for f in fails: st.write(f)
        if ui["autosave"]:
            save_store(ui["store_dir"], chunks, embs, idx)
            st.caption(f"Autosaved to {ui['store_dir']}")

# Save store
if st.session_state["index"] is not None and ui["do_save"]:
    save_store(ui["store_dir"], st.session_state["corpus_chunks"], st.session_state["embeddings"], st.session_state["index"])
    st.success(f"Saved store to {ui['store_dir']}")

# Tabs
with tab_chat:
    # choose role
    st.session_state["user_role"] = st.selectbox("Answer style (role)", 
        ["Association Analyst","Buyer / Feeder","Genetic Advisor","Independent Rancher"],
        index=["Association Analyst","Buyer / Feeder","Genetic Advisor","Independent Rancher"].index(
            st.session_state.get("user_role","Association Analyst")))
    render_tab_chat(st.session_state)

with tab_generate:
    render_generator(ui, st.session_state["corpus_chunks"], st.session_state["index"])

st.markdown("---")
st.caption("Tune chunk size/overlap; use filters; OCR for scanned PDFs; Autosave to persist.")