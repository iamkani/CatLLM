from __future__ import annotations
from typing import List, Sequence
import math

import numpy as np

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


def _encode_len(text: str, model: str | None = None) -> int:
    if not isinstance(text, str):
        text = str(text)
    if tiktoken is None:
        return max(1, len(text) // 4)
    try:
        enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding('cl100k_base')
    except Exception:
        enc = tiktoken.get_encoding('cl100k_base')
    return len(enc.encode(text))


def _split_on_boundaries(text: str, max_tokens: int, model: str | None = None) -> List[str]:
    if _encode_len(text, model) <= max_tokens:
        return [text]
    approx_chars = max_tokens * 4
    parts: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + approx_chars)
        window = text[i:j]
        cut = window.rfind('

')
        if cut >= approx_chars * 2 // 3:
            j = i + cut
            window = text[i:j]
        else:
            s_cut = window.rfind('. ')
            if s_cut >= approx_chars * 2 // 3:
                j = i + s_cut + 1
                window = text[i:j]
        while _encode_len(window, model) > max_tokens and len(window) > 0:
            shrink = max(1, len(window) // 10)
            window = window[:-shrink]
            j = i + len(window)
        if not window:
            break
        parts.append(window.strip())
        i = j
    out: List[str] = []
    for p in parts:
        while _encode_len(p, model) > max_tokens:
            ratio = max_tokens / max(1, _encode_len(p, model))
            take = max(1, int(len(p) * ratio))
            out.append(p[:take])
            p = p[take:]
        if p:
            out.append(p)
    return [s for s in out if s]


def embed_texts(client, texts: Sequence[str], embed_model: str, *, max_tokens_per_item: int = 7900, batch_size: int = 32):
    per_item_parts: List[List[str]] = []
    flat: List[str] = []
    for t in texts:
        if t is None:
            t = ''
        parts = _split_on_boundaries(str(t), max_tokens=max_tokens_per_item, model=embed_model)
        parts = [p for p in parts if isinstance(p, str) and p.strip()] or ['']
        per_item_parts.append(parts)
        flat.extend(parts)

    embs: List[List[float]] = []
    for s in range(0, len(flat), batch_size):
        batch = flat[s:s+batch_size]
        resp = client.embeddings.create(model=embed_model, input=batch)
        for item in resp.data:
            embs.append(item.embedding)

    pooled: List[List[float]] = []
    pos = 0
    for parts in per_item_parts:
        cnt = len(parts)
        if cnt == 1:
            pooled.append(embs[pos])
            pos += 1
            continue
        arr = np.array(embs[pos:pos+cnt], dtype=float)
        vec = arr.mean(axis=0)
        pooled.append(vec.tolist())
        pos += cnt

    return pooled
