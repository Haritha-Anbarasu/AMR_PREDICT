"""
Core utilities used across the entire pipeline: logging configuration,
a reusable exception hierarchy, and a timing/progress decorator.

Scientific/engineering rationale (for viva):
- Centralized logging (rather than scattered print() calls) is standard
  practice in reproducible bioinformatics pipelines — it produces an
  auditable run record (what ran, when, with what parameters, and what
  failed) that can be attached as supplementary material to a dissertation
  or publication, and is required if this pipeline is ever wrapped in
  Snakemake/Nextflow later (see PIPELINE_ARCHITECTURE notes in README).
- A custom exception hierarchy (rather than bare `except Exception`)
  lets calling code distinguish between "external tool missing",
  "malformed input data", and "model/feature mismatch" — each of which
  needs a different recovery action, and blanket exception handling
  would hide which failure mode actually occurred.
"""
import logging
import sys
import time
import functools
from pathlib import Path


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def _resolve_log_settings():
    """
    Reads log directory/filename/level from config.yaml when available,
    falling back to sane defaults so utils.py has no hard startup
    dependency on config.yaml existing (useful before the project is
    fully set up, or in quick scratch scripts).
    """
    try:
        from src.config import cfg
        log_dir = Path(cfg("paths.logs_dir", "logs"))
        log_file = cfg("logging.log_file", "pipeline.log")
        level_name = cfg("logging.level", "INFO")
        level = getattr(logging, str(level_name).upper(), logging.INFO)
        return log_dir, log_file, level
    except Exception:
        return Path("logs"), "pipeline.log", logging.INFO


LOG_DIR, _DEFAULT_LOG_FILE, _DEFAULT_LOG_LEVEL = _resolve_log_settings()


def get_logger(name: str, log_file: str = None, level=None) -> logging.Logger:
    """
    Returns a logger that writes to both console and a single execution
    log file at logs/pipeline.log (path configurable via config.yaml:
    paths.logs_dir / logging.log_file). Every pipeline step logs to this
    same file with timestamps, giving one complete, chronological
    execution record per run. Call this once per module:
        logger = get_logger(__name__)
    """
    log_file = log_file or _DEFAULT_LOG_FILE
    level = level if level is not None else _DEFAULT_LOG_LEVEL

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:  # avoid duplicate handlers on repeated calls / notebook re-runs
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_DIR / log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# ----------------------------------------------------------------------
# Exception hierarchy
# ----------------------------------------------------------------------
class AMRPredictError(Exception):
    """Base exception for all AMR-PREDICT pipeline errors."""


class ExternalToolError(AMRPredictError):
    """Raised when an external CLI tool (Prodigal, ABRicate, CD-HIT, ...) is
    missing or fails. Distinct from data errors — the fix here is
    installation, not data cleaning."""


class DataFormatError(AMRPredictError):
    """Raised when input/intermediate data does not match the expected
    schema (e.g. a FASTA file with no headers, a labels CSV missing a
    required column). Distinct from tool errors — the fix here is
    upstream data validation."""


class ModelError(AMRPredictError):
    """Raised for model training/prediction failures, e.g. feature
    mismatch between training and inference, or an untrained model
    being used for prediction."""


# ----------------------------------------------------------------------
# Timing / progress decorator
# ----------------------------------------------------------------------
def timed_step(step_name: str = None):
    """
    Decorator that logs start/end/duration of a pipeline step, and
    re-raises any exception after logging it (so the log has full
    traceback context but the caller's error handling still runs).

    Usage:
        @timed_step("Gene prediction")
        def predict_genes(...): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            label = step_name or func.__name__
            logger.info(f"[START] {label}")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(f"[DONE]  {label} ({elapsed:.1f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"[FAILED] {label} after {elapsed:.1f}s — {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator


def get_progress_bar(iterable=None, total=None, desc=""):
    """
    Thin wrapper around tqdm so the rest of the codebase doesn't need to
    import tqdm directly (and degrades gracefully if tqdm isn't installed).
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc)
    except ImportError:
        # Fallback: no progress bar, just return the iterable
        return iterable
