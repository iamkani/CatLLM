# catllm/embeddings.py
from typing import Iterable, List
import json
import logging
import numpy as np
import math
import re
import time
import unicodedata
from openai import BadRequestError

logger = logging.getLogger(__name__)

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")  # strip lone surrogates (invalid in JSON)

def _normalize_text(s: str) -> str:
    # 1) normalize unicode  2) drop surrogates  3) strip control chars except \t\r\n
    s = unicodedata.normalize("NFC", s)
    s = _SURROGATE_RE.sub("�", s)                     # replace with U+FFFD
    s = "".join(ch for ch in s if (ch >= " " or ch in "\t\r\n"))
    return s.strip()

def _to_clean_str(x):
    if x is None:
        return None
    # bytes → str
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", errors="replace")
        except Exception:
            return None
    # numpy scalars, objects, etc. → str
    try:
        x = str(x)
    except Exception:
        return None
    x = _normalize_text(x)
    if not x:
        return None
    # 8k chars cap per item (avoid huge JSON & token overruns)
    return x[:8000]

def _as_list(texts: Iterable) -> List:
    # turn generators / numpy arrays / pandas Series into a list
    try:
        if hasattr(texts, "tolist"):
            return list(texts.tolist())
    except Exception:
        pass
    return list(texts)

def _sanitize_inputs(texts: Iterable) -> List[str]:
    items = _as_list(texts)
    cleaned: List[str] = []
    dropped = 0
    for it in items:
        s = _to_clean_str(it)
        if s is None:
            dropped += 1
            continue
        cleaned.append(s)
    # local JSON validity check; disallow NaN/Infinity
    try:
        json.dumps({"input": cleaned}, ensure_ascii=True, allow_nan=False)
    except Exception as e:
        # Find the first offender more precisely by scanning items
        for idx, s in enumerate(cleaned):
            try:
                json.dumps(s, ensure_ascii=True, allow_nan=False)
            except Exception:
                raise ValueError(f"Bad JSON string at index {idx}: {repr(s[:120])} ...") from e
        # Fallback
        raise
    if not cleaned:
        raise ValueError("No valid texts to embed after sanitization.")
    return cleaned

def _batched(seq: List[str], size: int):
    for i in range(0, len(seq), size):
        yield i, seq[i:i+size]

def embed_texts(
    client,
    texts: Iterable,
    model: str,
    batch_size: int = 128,
    max_req_tokens: int = 7000,
    sleep_between: float = 0.1,
):
    """
    Returns a numpy array of embeddings (N, dim) aligned with the input order.

    Combines sanitisation with token-safe batching so that individual API
    requests stay within safe token limits.
    """
    safe = _sanitize_inputs(texts)
    all_embs: List[List[float]] = []

    # Token-safe batching: respect both item count and approximate token budget
    i = 0
    while i < len(safe):
        batch: List[str] = []
        tok_sum = 0
        while i < len(safe) and len(batch) < batch_size:
            tks = max(1, len(safe[i]) // 4)  # ≈ 4 chars per token
            if batch and tok_sum + tks > max_req_tokens:
                break
            batch.append(safe[i])
            tok_sum += tks
            i += 1

        # Final per-batch JSON validity check
        json.dumps({"input": batch}, ensure_ascii=True, allow_nan=False)
        try:
            resp = client.embeddings.create(model=model, input=batch)
        except BadRequestError as e:
            # Narrow down which item in this batch breaks things
            offset = i - len(batch)
            for j, s in enumerate(batch):
                try:
                    client.embeddings.create(model=model, input=[s])
                except BadRequestError:
                    raise ValueError(
                        f"Embeddings batch failed. First bad item at global index {offset + j}. "
                        f"Snippet: {repr(s[:160])}"
                    ) from e
            raise
        all_embs.extend(d.embedding for d in resp.data)
        if sleep_between > 0:
            time.sleep(sleep_between)

    logger.debug("Embedded %d texts into %d vectors", len(safe), len(all_embs))
    return np.asarray(all_embs, dtype="float32")


def embed_texts_cached(
    client,
    texts: Iterable,
    model: str,
    cache: dict = None,
    **kwargs,
) -> np.ndarray:
    """Thin wrapper around embed_texts with an optional dict cache for small queries.

    Only caches batches of ≤ 2 texts (i.e. single-query lookups).
    Corpus-scale calls pass through uncached.
    """
    text_list = list(texts) if not isinstance(texts, list) else texts
    if cache is None or len(text_list) > 2:
        return embed_texts(client, text_list, model, **kwargs)

    key = (model, tuple(text_list))
    if key in cache:
        logger.debug("Embedding cache hit for %d texts", len(text_list))
        return cache[key]

    result = embed_texts(client, text_list, model, **kwargs)
    cache[key] = result
    return result