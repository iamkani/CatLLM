# catllm/indexer.py
"""
FAISS index build & search utilities for CatLLM.

Exports:
- build_faiss_index(embeddings, metric="ip", kind="flat", **kwargs)
- top_k_search(index, query_vec, k=5)
- batch_search(index, query_mat, k=5)
- normalize_l2(x)
"""

from typing import Tuple, Optional
import numpy as np
import faiss


def normalize_l2(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D array (in-place safe copy)."""
    if x.ndim != 2:
        raise ValueError("normalize_l2 expects a 2D array")
    y = x.astype("float32", copy=True)
    faiss.normalize_L2(y)
    return y


def _metric_to_faiss(metric: str) -> int:
    m = metric.lower()
    if m in ("ip", "inner", "cosine"):  # cosine == IP after L2 norm
        return faiss.METRIC_INNER_PRODUCT
    if m in ("l2", "euclidean"):
        return faiss.METRIC_L2
    raise ValueError(f"Unsupported metric: {metric}. Use 'ip' or 'l2'.")


def build_faiss_index(
    embeddings: np.ndarray,
    metric: str = "ip",
    kind: str = "flat",
    # IVF/HNSW params (ignored for flat unless specified)
    nlist: int = 0,
    hnsw_m: int = 32,
    ef_construction: int = 100,
    ef_search: Optional[int] = None,
) -> faiss.Index:
    """
    Build a FAISS index from dense vectors.

    Args:
      embeddings: shape (N, D) float32 array.
      metric: "ip" (default) or "l2". For cosine similarity, use "ip" and L2-normalize vectors.
      kind: "flat" (default), "ivf-flat", or "hnsw".
      nlist: number of IVF clusters (e.g., 1024) if kind="ivf-flat".
      hnsw_m: HNSW graph degree (kind="hnsw").
      ef_construction: HNSW construction parameter (kind="hnsw").
      ef_search: HNSW search parameter; if given we set it on returned index.

    Returns:
      A FAISS index with the embeddings added.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D [N, D]")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    faiss_metric = _metric_to_faiss(metric)

    # Cosine = inner product on L2-normalized vectors
    if metric.lower() in ("ip", "cosine"):
        base_vecs = normalize_l2(embeddings)
    else:
        base_vecs = embeddings

    kind = kind.lower()

    if kind == "flat":
        if faiss_metric == faiss.METRIC_INNER_PRODUCT:
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)

    elif kind == "ivf-flat":
        if nlist <= 0:
            # heuristic: sqrt(N) capped
            n = base_vecs.shape[0]
            nlist = max(64, min(4096, int(np.sqrt(max(1, n)))))
        quantizer = faiss.IndexFlatIP(dim) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss_metric)
        # train then add
        index.train(base_vecs)
    elif kind == "hnsw":
        # HNSW inner product uses IndexHNSWFlat + METRIC_INNER_PRODUCT
        # For L2, we’ll use the same and rely on metric at search time
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss_metric)
        index.hnsw.efConstruction = ef_construction
        if ef_search is not None:
            index.hnsw.efSearch = ef_search
    else:
        raise ValueError(f"Unsupported index kind: {kind} (use 'flat', 'ivf-flat', or 'hnsw')")

    # Add vectors to index
    index.add(base_vecs)
    return index


def _ensure_query_2d(q: np.ndarray) -> np.ndarray:
    if q.ndim == 1:
        q = q[None, :]
    if q.dtype != np.float32:
        q = q.astype("float32")
    return q


def top_k_search(index: faiss.Index, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search a single query vector against the index.

    Args:
      index: FAISS index
      query_vec: shape (D,) or (1, D)
      k: number of neighbors

    Returns:
      (indices, scores) as 1D arrays of length k
    """
    q = _ensure_query_2d(query_vec)
    # If index expects cosine/IP, normalize queries as well
    if isinstance(index, (faiss.IndexFlatIP, faiss.IndexIVFFlat, faiss.IndexHNSWFlat)):
        # Not all subclasses expose metric publicly; safest is to L2-normalize
        faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return I[0], D[0]


def batch_search(index: faiss.Index, query_mat: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search multiple queries at once.

    Args:
      index: FAISS index
      query_mat: shape (Q, D)
      k: neighbors

    Returns:
      (I, D) where I/D are (Q, k)
    """
    q = _ensure_query_2d(query_mat)
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return I, D

# --- Simple search wrapper expected by UI ---

import numpy as _np
import faiss as _faiss

def _mmr_reorder(
    candidate_indices,
    candidate_scores,
    query_vec,
    embeddings,
    lambda_mult: float = 0.5,
    top_k: int = 5,
):
    """
    Maximal Marginal Relevance (very light) to diversify results.
    Expects:
      - candidate_indices: 1D np.array of indices from FAISS
      - candidate_scores:  1D np.array of similarity scores (IP)
      - query_vec:         2D np.array shape (1, dim)
      - embeddings:        2D np.array shape (N, dim) normalized
    Returns: reordered list of indices (length <= top_k)
    """
    if embeddings is None or len(candidate_indices) == 0:
        return candidate_indices[:top_k].tolist()

    # Normalize query (defensive)
    q = query_vec.astype("float32").copy()
    _faiss.normalize_L2(q)

    selected = []
    candidates = candidate_indices.tolist()

    # Pre-pull candidate vectors
    cand_vecs = embeddings[candidates].astype("float32")
    _faiss.normalize_L2(cand_vecs)

    # Similarity to query (cosine since normalized)
    sim_to_q = cand_vecs @ q[0]

    # Greedy MMR
    while len(selected) < min(top_k, len(candidates)):
        best_idx = -1
        best_score = -1e9
        for pos, global_i in enumerate(candidates):
            if global_i in selected:
                continue
            # redundancy: max similarity to already selected
            if selected:
                sel_vecs = embeddings[selected].astype("float32")
                _faiss.normalize_L2(sel_vecs)
                red = (cand_vecs[pos][None, :] @ sel_vecs.T).max()
            else:
                red = 0.0
            score = lambda_mult * sim_to_q[pos] - (1 - lambda_mult) * red
            if score > best_score:
                best_score = score
                best_idx = global_i
        selected.append(best_idx)
        # remove from candidates list
        if best_idx in candidates:
            rm_pos = candidates.index(best_idx)
            candidates.pop(rm_pos)
            cand_vecs = _np.delete(cand_vecs, rm_pos, axis=0)
            sim_to_q = _np.delete(sim_to_q, rm_pos, axis=0)

    return selected

def search(
    index,
    query_vec: _np.ndarray,
    k: int = 5,
    diversify: bool = False,
    lambda_mult: float = 0.5,
    embeddings: _np.ndarray | None = None,
):
    """
    Wrapper expected by UI:
      - Runs FAISS search
      - Optionally applies simple MMR diversification
    Returns: (indices: List[int], scores: List[float])
    """
    # Base search (get a few extra if diversifying)
    base_k = max(k, 1)
    raw_k = base_k * 5 if diversify else base_k

    I, D = top_k_search(index, query_vec, k=raw_k)  # I: indices, D: scores

    if diversify:
        if embeddings is None:
            # no embeddings to diversify; just return top-k
            return I[:k].tolist(), D[:k].tolist()
        reordered = _mmr_reorder(
            candidate_indices=_np.array(I),
            candidate_scores=_np.array(D),
            query_vec=query_vec,
            embeddings=embeddings,
            lambda_mult=lambda_mult,
            top_k=k,
        )
        # map reordered back to scores by position
        idx_to_score = {int(i): float(s) for i, s in zip(I, D)}
        return reordered, [idx_to_score[int(i)] for i in reordered]

    # no diversification
    return I[:k].tolist(), D[:k].tolist()