from typing import List, Tuple, Dict, Callable
from .utils_text import read_docs, read_local_folder

def ingest_uploads(files, **kwargs) -> List[Tuple[str, str]]:
    # kwargs: use_ocr, ocr_max_pages, ocr_dpi, seen_hashes
    return read_docs(files, **kwargs)

def ingest_folder(path: str, progress_cb: Callable | None = None, **kwargs):
    # kwargs: exts, recursive, max_files, max_size_mb, use_ocr, ocr_max_pages, ocr_dpi, seen_hashes
    return read_local_folder(path, progress_cb=progress_cb, **kwargs)