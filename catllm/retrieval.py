# catllm/retrieval.py
from __future__ import annotations

import faiss  # type: ignore
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

from .tagging import tag_text_for_meta, expand_query, normalize_cluster_name
from .utils_text import chunk_text


# -----------------------------
# FAISS helpers
# -----------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a cosine-sim (L2-normalized + inner product) FAISS index.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array (N, dim)")
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    idx.add(embeddings.astype("float32"))
    return idx


def search_index(
    index: faiss.Index,
    query_vecs: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for batched queries.
    Returns (I, D) with shapes (B, k).
    """
    q = query_vecs.astype("float32").copy()
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return I, D


# -----------------------------
# Corpus assembly
# -----------------------------
def make_chunks_from_docs(
    docs: Sequence[Tuple[str, str, Dict]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> List[Dict]:
    """
    Convert (title, text, meta) docs into chunk dicts:
      {"text": <chunk>, "meta": {title, source, cluster?, path?, breeds/traits/genes}}
    """
    chunks: List[Dict] = []
    for title, text, extra in docs:
        for ch in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            meta = {"title": title, "source": title}
            if isinstance(extra, dict):
                meta.update(extra)  # cluster, path, etc.
            meta.update(tag_text_for_meta(ch))
            chunks.append({"text": ch, "meta": meta})
    return chunks


# -----------------------------
# Filtering utilities
# -----------------------------
def filter_candidates_by_cluster(
    candidates: List[Dict],
    selected_clusters: Sequence[str],
    top_k: int,
) -> List[Dict]:
    if not selected_clusters:
        return candidates
    norm_selected = {normalize_cluster_name(c) for c in selected_clusters}
    filtered = [
        c for c in candidates
        if normalize_cluster_name(c.get("meta", {}).get("cluster", "")) in norm_selected
    ]
    return filtered if len(filtered) >= top_k else candidates


def filter_candidates_by_tags(
    candidates: List[Dict],
    breeds: Sequence[str],
    traits: Sequence[str],
    genes: Sequence[str],
    top_k: int,
) -> List[Dict]:
    if not (breeds or traits or genes):
        return candidates

    def _ok(meta: Dict) -> bool:
        if breeds and not (set(meta.get("breeds", [])) & set(map(str.lower, breeds))):
            return False
        if traits and not (set(meta.get("traits", [])) & set(map(str.lower, traits))):
            return False
        if genes and not (set(meta.get("genes", [])) & set(map(str.lower, genes))):
            return False
        return True

    filtered = [c for c in candidates if _ok(c.get("meta", {}))]
    return filtered if len(filtered) >= top_k else candidates


# -----------------------------
# Context formatter
# -----------------------------
def format_context(chunks: Sequence[Dict], max_chars: int = 1200) -> str:
    """
    Produce a compact context string with numbered items the model can cite as [#].
    """
    parts: List[str] = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("meta", {})
        title = meta.get("title", "Unknown")
        src = meta.get("path", "") or meta.get("source", "")
        excerpt = (c.get("text", "") or "")[:max_chars]
        parts.append(f"[{i}] Title: {title}\nURL/Source: {src}\nExcerpt:\n{excerpt}")
    return "\n\n---\n\n".join(parts)


# -----------------------------
# End-to-end retrieve
# -----------------------------
def retrieve_context_for_query(
    *,
    client,
    embed_model: str,
    index: faiss.Index,
    corpus_chunks: Sequence[Dict],
    user_query: str,
    use_synonyms: bool = True,
    top_k: int = 5,
    selected_clusters: Optional[Sequence[str]] = None,
    selected_breeds: Optional[Sequence[str]] = None,
    selected_traits: Optional[Sequence[str]] = None,
    selected_genes: Optional[Sequence[str]] = None,
) -> Tuple[str, List[Dict], List[int], List[float]]:
    """
    1) Expand query with synonyms (optional).
    2) Embed query.
    3) Retrieve K' (K or larger if filters set) candidates.
    4) Apply cluster/tag filters without starving results.
    5) Return formatted context and the selected chunks.

    Returns:
      ctx_text, ctx_items, indices, scores
    """
    selected_clusters = selected_clusters or []
    selected_breeds = selected_breeds or []
    selected_traits = selected_traits or []
    selected_genes = selected_genes or []

    q = expand_query(user_query) if use_synonyms else user_query
    q_vec = _embed_small(client, embed_model, [q])

    # If filters are active, retrieve more to avoid empty results post-filter.
    search_k = top_k
    if selected_clusters or selected_breeds or selected_traits or selected_genes:
        search_k = max(top_k * 5, 50)

    I, D = search_index(index, q_vec, k=search_k)
    idxs = I[0].tolist()
    scs = D[0].tolist()
    candidates = [corpus_chunks[i] for i in idxs]

    # Filter by cluster/tag (but don't starve if too few remain)
    candidates = filter_candidates_by_cluster(candidates, selected_clusters, top_k)
    candidates = filter_candidates_by_tags(
        candidates, selected_breeds, selected_traits, selected_genes, top_k
    )

    # Take top_k post-filter
    chosen = candidates[:top_k]

    # Rebuild indices/scores aligned to chosen (best effort)
    chosen_indices: List[int] = []
    chosen_scores: List[float] = []
    seen = set()
    for c in chosen:
        try:
            j = corpus_chunks.index(c)
        except ValueError:
            # fallback: find first with same title & text
            j = -1
            meta = c.get("meta", {})
            for k, cc in enumerate(corpus_chunks):
                if k in seen:
                    continue
                if cc.get("text") == c.get("text") and cc.get("meta", {}).get("title") == meta.get("title"):
                    j = k
                    break
        if j >= 0:
            seen.add(j)
            chosen_indices.append(j)
            # map score if present
            if j in idxs:
                chosen_scores.append(scs[idxs.index(j)])
            else:
                chosen_scores.append(0.0)

    ctx_text = format_context(chosen)
    return ctx_text, chosen, chosen_indices, chosen_scores


# -----------------------------
# Small embed helper
# -----------------------------
def _embed_small(client, embed_model: str, input_list: Sequence[str]) -> np.ndarray:
    """
    Minimal helper to embed a small list (<= 16 items).
    """
    if not input_list:
        return np.zeros((0, 1), dtype="float32")
    r = client.embeddings.create(model=embed_model, input=list(input_list))
    return np.array([d.embedding for d in r.data], dtype="float32")
