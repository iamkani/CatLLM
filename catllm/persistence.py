import json, pathlib, numpy as np, faiss
from typing import List, Dict, Tuple, Any

def save_store(store_dir: str, corpus_chunks: List[Dict], embeddings: np.ndarray, index: Any) -> None:
    p = pathlib.Path(store_dir); p.mkdir(parents=True, exist_ok=True)
    with open(p/"corpus.json","w",encoding="utf-8") as f:
        json.dump(corpus_chunks, f, ensure_ascii=False)
    np.save(p/"embeddings.npy", embeddings)
    faiss.write_index(index, str(p/"faiss.index"))

def load_store(store_dir: str):
    p = pathlib.Path(store_dir)
    if not (p/"corpus.json").exists(): return None, None, None
    with open(p/"corpus.json","r",encoding="utf-8") as f:
        corpus_chunks = json.load(f)
    embeddings = np.load(p/"embeddings.npy")
    index = faiss.read_index(str(p/"faiss.index"))
    return corpus_chunks, embeddings, index