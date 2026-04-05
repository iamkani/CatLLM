# catllm/utils_text.py
import os
import io
import re
import hashlib
import logging
from typing import List, Tuple, Dict, Callable, Optional

from pypdf import PdfReader

logger = logging.getLogger(__name__)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# -----------------------
# Synonyms / query expand
# -----------------------
SYNONYMS: Dict[str, List[str]] = {
    "ebv": ["epd", "estimated breeding value"],
    "epd": ["ebv", "expected progeny difference"],
    "mstn": ["myostatin"],
    "dgat1": ["k232a"],
    "marbling": ["intramuscular fat", "imf"],
    "calving": ["calving ease", "ce"],
    "heritability": ["h2", "h²"],
    "snps": ["snp", "single nucleotide polymorphism"],
    "qtl": ["quantitative trait locus"],
}

def approx_token_count(s: str) -> int:
    """
    Very rough token heuristic (≈ 4 chars per token). Returns at least 1.
    Useful for batching to stay under model context limits.
    """
    if not s:
        return 1
    return max(1, len(s) // 4)

def expand_query(q: str) -> str:
    q_low = q.lower()
    extra_terms: List[str] = []
    tokens = set([t.strip(".,;:!?()[]{}\"'") for t in q_low.split()])
    for tok in list(tokens):
        if tok in SYNONYMS:
            extra_terms.extend(SYNONYMS[tok])
    for key, syns in SYNONYMS.items():
        if " " in key and key in q_low:
            extra_terms.extend(syns)
    extra_terms = list(dict.fromkeys(extra_terms))
    if not extra_terms:
        return q
    return q + " " + " ".join(extra_terms)


def looks_like_idk(text: str) -> bool:
    """Check if an LLM answer declares it doesn't know."""
    t = (text or "").lower()
    return any(
        p in t
        for p in [
            "i don't know",
            "i do not know",
            "not present in the context",
            "cannot find this in the context",
            "no information in the provided context",
        ]
    )


# -------------
# Text chunking
# -------------
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """Simple word-window chunker."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


_PARAGRAPH_RE = re.compile(r'\n\s*\n')
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def chunk_text_smart(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """Paragraph-aware chunker: splits on paragraph boundaries, then sentences,
    falling back to word-window for oversized blocks."""
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
    if not paragraphs:
        return chunk_text(text, chunk_size, overlap)

    # Break large paragraphs into sentences
    units: List[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= chunk_size:
            units.append(para)
        else:
            sentences = _SENTENCE_RE.split(para)
            for sent in sentences:
                sw = sent.split()
                if len(sw) <= chunk_size:
                    units.append(sent)
                else:
                    # Oversized sentence — word-window fallback
                    units.extend(chunk_text(sent, chunk_size, overlap))

    # Accumulate units into chunks respecting size limit
    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0

    for unit in units:
        uw = len(unit.split())
        if buf and buf_words + uw > chunk_size:
            chunks.append("\n\n".join(buf))
            # Carry overlap from end of emitted chunk
            carry = " ".join("\n\n".join(buf).split()[-overlap:]) if overlap > 0 else ""
            buf = [carry] if carry else []
            buf_words = len(carry.split()) if carry else 0
        buf.append(unit)
        buf_words += uw

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks if chunks else chunk_text(text, chunk_size, overlap)

# ----------------
# Deduping helpers
# ----------------
def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

# ----
# OCR
# ----
def ocr_pdf_bytes(pdf_bytes: bytes, max_pages: int = 10, dpi: int = 200) -> str:
    """
    Optional OCR fallback using pdf2image + pytesseract.
    Requires system deps:
      - Poppler (for pdf2image)
      - Tesseract OCR
    If not installed, returns empty string.
    """
    try:
        import pdf2image
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError:
        logger.debug("OCR dependencies not installed (pdf2image/pytesseract/Pillow)")
        return ""
    try:
        pages = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
        texts: List[str] = []
        for img in pages:
            try:
                txt = pytesseract.image_to_string(img)
            except Exception:
                logger.warning("OCR failed on a page", exc_info=True)
                txt = ""
            if txt:
                texts.append(txt)
        return "\n".join(texts)
    except Exception:
        logger.warning("OCR conversion failed", exc_info=True)
        return ""

# ---------------
# File ingest I/O
# ---------------
def read_docs(
    uploaded_files,
    use_ocr: bool = False,
    ocr_max_pages: int = 10,
    ocr_dpi: int = 200,
    seen_hashes: Optional[set] = None,
) -> List[Tuple[str, str]]:
    """
    Read a list of Streamlit-uploaded files. Returns [(title, text)].
    Deduplicates by SHA-256 of raw bytes. OCR fallback for PDFs if requested.
    """
    if seen_hashes is None:
        seen_hashes = set()
    docs: List[Tuple[str, str]] = []

    for f in uploaded_files:
        # Read raw bytes
        try:
            raw = f.read()
        except Exception:
            raw = f.read()
        if not isinstance(raw, (bytes, bytearray)):
            try:
                raw = bytes(raw)
            except Exception:
                raw = b""

        # Dedupe
        file_hash = _sha256_bytes(raw)
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)

        name = f.name
        suffix = name.lower().split(".")[-1] if "." in name else ""
        text = ""

        if suffix == "pdf":
            # Try text extraction first
            try:
                reader = PdfReader(io.BytesIO(raw), strict=False)
                pieces = []
                for page in reader.pages:
                    pieces.append(page.extract_text() or "")
                text = "\n".join(pieces)
            except Exception:
                logger.warning("PDF text extraction failed for %s", name, exc_info=True)
                text = ""

            # Optional OCR if no text
            if use_ocr and not text.strip():
                text = ocr_pdf_bytes(raw, max_pages=ocr_max_pages, dpi=ocr_dpi)

            # Final fallback: decode bytes
            if not text:
                try:
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    text = raw.decode("latin-1", errors="ignore")
        else:
            # Non-PDF: decode bytes
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                text = raw.decode("latin-1", errors="ignore")

        docs.append((name, text))

    return docs

def read_local_folder(
    folder_path: str,
    exts: Optional[List[str]] = None,
    recursive: bool = True,
    max_files: int = 200,
    max_size_mb: int = 50,
    use_ocr: bool = False,
    ocr_max_pages: int = 10,
    ocr_dpi: int = 200,
    seen_hashes: Optional[set] = None,
    progress_cb: Optional[Callable[[int, int, str, str], None]] = None,
):
    """
    Read local files with dedupe + optional OCR.
    Returns (docs, fails) where:
      docs = [(title, text, {"cluster": <subfolder>, "path": <fullpath>})]
      fails = [list of file paths that failed]
    progress_cb(i, total, path, phase) where phase in {"scan","parse","done"}.
    """
    import pathlib as _pl

    if seen_hashes is None:
        seen_hashes = set()
    if exts is None:
        exts = ["pdf", "txt", "md", "csv"]

    root = _pl.Path(folder_path).expanduser().resolve()
    docs: List[Tuple[str, str, Dict[str, str]]] = []
    fails: List[str] = []

    if not (root.exists() and root.is_dir()):
        return [], []

    max_bytes = max_size_mb * 1024 * 1024

    # Build candidate list
    paths = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                paths.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                paths.append(p)

    # Trim & deterministic order
    paths = sorted(paths)[:max_files]
    total = len(paths)

    for i, p in enumerate(paths, 1):
        if progress_cb:
            progress_cb(i, total, str(p), "scan")

        try:
            if p.stat().st_size > max_bytes:
                continue

            with open(p, "rb") as fh:
                raw = fh.read()

            file_hash = _sha256_bytes(raw)
            if file_hash in seen_hashes:
                continue
            seen_hashes.add(file_hash)

            if progress_cb:
                progress_cb(i, total, str(p), "parse")

            text = ""
            if p.suffix.lower() == ".pdf":
                try:
                    reader = PdfReader(io.BytesIO(raw), strict=False)
                    pieces = []
                    for page in reader.pages:
                        pieces.append(page.extract_text() or "")
                    text = "\n".join(pieces)
                except Exception:
                    logger.warning("PDF text extraction failed for %s", p, exc_info=True)
                    text = ""
                if use_ocr and not text.strip():
                    text = ocr_pdf_bytes(raw, max_pages=ocr_max_pages, dpi=ocr_dpi)
                if not text:
                    try:
                        text = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        text = raw.decode("latin-1", errors="ignore")
            else:
                try:
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    text = raw.decode("latin-1", errors="ignore")

            # derive cluster (first-level subfolder relative to root)
            try:
                rel_parent = p.parent.resolve().relative_to(root)
                rel_str = str(rel_parent)
                cluster = rel_str.split(os.sep)[0] if rel_str else ""
            except Exception:
                cluster = ""

            from .tagging import normalize_cluster_name
            cluster = normalize_cluster_name(cluster)

            docs.append((p.name, text, {"cluster": cluster, "path": str(p)}))

        except Exception:
            logger.warning("Failed to process %s", p, exc_info=True)
            fails.append(str(p))
            continue

        if progress_cb:
            progress_cb(i, total, str(p), "done")

    return docs, fails