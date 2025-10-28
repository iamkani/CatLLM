import logging, os

def setup_logging(level: str | int = None):
    if level is None:
        level = os.getenv("CATLLM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("catllm")