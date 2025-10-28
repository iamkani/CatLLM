import json, streamlit as st, pandas as pd, numpy as np
from ..embeddings import embed_texts
from ..indexer import search
from ..llm import ensure_client, synthesize_answer
from ..retrieval import format_context

def render_generator(state, corpus_chunks, index):
    st.subheader("Generate Q&A from CSV/XLSX using your current RAG index")
    if index is None or not corpus_chunks:
        st.info("Load or build an index first.")
        return

    qa_file = st.file_uploader("Upload CSV or XLSX with questions (and optional answers)", type=["csv","xlsx","xls"])
    infer_topk = st.slider("Top-K passages per question", 1, 10, state.get("infer_topk",5), 1)
    run_btn = st.button("Generate Q&A")

    if not (run_btn and qa_file): return

    # read CSV/XLSX
    if qa_file.name.lower().endswith((".xlsx",".xls")):
        try: import openpyxl  # noqa
        except Exception: st.error("Please install openpyxl: pip install openpyxl"); return
        df = pd.read_excel(qa_file)
    else:
        df = pd.read_csv(qa_file)

    q_col = next((c for c in df.columns if c.lower() in {"question","q","prompt","query"}), None)
    if not q_col: st.error("Could not find a question column (question/q/prompt/query)."); return

    client = ensure_client(state["provider"])
    questions = [str(x).strip() for x in df[q_col].tolist() if str(x).strip()]
    progress = st.progress(0); results = []; total = len(questions)

    for i, qtext in enumerate(questions, 1):
        qvec = embed_texts(client, state["embed_model"], [qtext])
        I, D = search(index, qvec, k=infer_topk)
        ctx_items = [corpus_chunks[j] for j in I]
        ctx_text = format_context(ctx_items)

        messages = [
            {"role":"system","content": state.get("system_prompt","You are a cattle-genetics RAG assistant.")},
            {"role":"user","content": f"QUESTION:\n{qtext}\n\nCONTEXT:\n{ctx_text}"}
        ]
        ans = synthesize_answer(client, state["provider"], state["chat_model"], messages)
        titles = [c.get("meta",{}).get("title","Unknown") for c in ctx_items]
        paths  = [c.get("meta",{}).get("path","") for c in ctx_items]

        results.append({"question": qtext, "answer": ans, "citations": "; ".join(titles), "doc_paths": "; ".join(paths)})

        if (i % 5 == 0) or (i == total):
            progress.progress(int(i/max(1,total)*100))

    out_df = pd.DataFrame(results)
    st.success(f"Generated {len(results)} Q&A rows.")
    st.dataframe(out_df, use_container_width=True)
    st.download_button("Download CSV", data=out_df.to_csv(index=False).encode("utf-8"),
                       file_name="generated_qa.csv", mime="text/csv")
    st.download_button("Download JSONL",
                       data="\n".join([json.dumps(r, ensure_ascii=False) for r in results]).encode("utf-8"),
                       file_name="generated_qa.jsonl", mime="application/json")
