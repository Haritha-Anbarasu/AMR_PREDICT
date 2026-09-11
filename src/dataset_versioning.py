"""
src/dataset_versioning.py

Records dataset provenance for every source database used by the
pipeline: which database, what version/download date, how many
sequences came in, and how many survived preprocessing (deduplication,
length filtering, etc.).

Why this matters for a dissertation: ARG databases are updated on their
own release schedules (CARD alone has had multiple versions with
materially different sequence counts). Without a recorded download
date and before/after counts, the exact dataset used cannot be
reconstructed or audited later — which is exactly the kind of detail
examiners and reviewers ask about. This module makes that record
automatic rather than relying on manual note-taking.

Output: reports/dataset_versions.csv (append-only — one row per
recorded event, so re-running data acquisition scripts builds a full
history rather than overwriting the previous record).
"""
import csv
from pathlib import Path
from datetime import datetime

from src.config import cfg

FIELDNAMES = [
    "timestamp", "source_db", "db_version", "download_date",
    "num_downloaded", "num_retained_after_preprocessing",
    "retention_pct", "notes",
]


def _version_log_path() -> Path:
    reports_dir = Path(cfg("paths.reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / "dataset_versions.csv"


def record_dataset_version(source_db: str, num_downloaded: int, num_retained: int,
                            db_version: str = "unversioned (live bulk download)",
                            notes: str = "") -> Path:
    """
    Append one provenance record. Call this at the end of every
    database-specific download/preprocessing step.

    Many ARG databases (CARD, MEGARes, ResFinder, AMRFinderPlus) don't
    expose a single clean version string through their bulk-download
    endpoints — where a real version tag isn't available, db_version
    defaults to a note that the download_date itself is the operative
    provenance marker (standard practice when citing "live" reference
    databases; record the date you pulled the data, not just the tool
    version).
    """
    path = _version_log_path()
    file_exists = path.exists()

    retention_pct = round((num_retained / num_downloaded) * 100, 2) if num_downloaded else 0.0

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_db": source_db,
            "db_version": db_version,
            "download_date": datetime.now().strftime("%Y-%m-%d"),
            "num_downloaded": num_downloaded,
            "num_retained_after_preprocessing": num_retained,
            "retention_pct": retention_pct,
            "notes": notes,
        })

    return path


def summarize_dataset_versions() -> str:
    """Returns a formatted string summary of the full version log — handy
    for printing at the end of a data-build script or embedding in a report."""
    path = _version_log_path()
    if not path.exists():
        return "No dataset version records yet."

    with open(path) as f:
        rows = list(csv.DictReader(f))

    lines = [f"{'Source DB':<18}{'Version':<30}{'Downloaded':<12}{'Retained':<12}{'Retention %':<12}"]
    lines.append("-" * 84)
    for row in rows:
        lines.append(
            f"{row['source_db']:<18}{row['db_version'][:28]:<30}"
            f"{row['num_downloaded']:<12}{row['num_retained_after_preprocessing']:<12}"
            f"{row['retention_pct']:<12}"
        )
    return "\n".join(lines)
