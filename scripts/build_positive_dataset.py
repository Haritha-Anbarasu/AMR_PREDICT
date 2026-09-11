"""
scripts/build_positive_dataset.py

Builds the final POSITIVE (ARG) training set:

    download_multi_database.py (CARD + AMRFinderPlus + ResFinder + MEGARes)
        -> standardize FASTA headers to one common schema
        -> merge into a single FASTA
        -> CD-HIT at 90% identity to remove duplicates/near-duplicates
        -> clean labeled positive dataset (FASTA + CSV)

Why standardize headers before merging?
Each database uses its own header convention (CARD: 'gb|ACC|ARO:xxxx|gene',
MEGARes: 'MEG_1|Drugs|class|mechanism|group', ResFinder: plain gene names,
AMRFinderPlus: RefSeq/GenBank accessions). Feeding these directly into
CD-HIT is fine (CD-HIT only clusters by sequence), but keeping headers
in four different formats downstream makes every later join brittle and
undocumented. We standardize to:

    >{source_db}__{original_id}  gene={gene_symbol}|class={drug_class}

so any downstream script can parse the header with one regex regardless
of origin database.

Why CD-HIT at 90% identity specifically?
This is the threshold most commonly used in ARG-database curation
literature (e.g. CARD's own protein homolog models, MEGARes clustering)
as a practical balance: below ~90%, sequence variants that ARE
functionally distinct alleles risk being collapsed into one cluster;
above ~95%, near-identical alleles across databases (the same gene
independently curated by CARD and ResFinder) are not merged, artificially
inflating class size and causing train/test leakage later. 90% is the
standard default and is defensible as a stated methodological choice in
a dissertation.

Usage:
    python scripts/build_positive_dataset.py --outdir data/training/positive
"""
import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, timed_step, ExternalToolError, DataFormatError
from src.dataset_versioning import record_dataset_version
from src.config import cfg

logger = get_logger(__name__)

HEADER_PATTERN = re.compile(r"^(\w+)__(.+)$")


def _label_dict_from_csv(labels_csv: Path) -> dict:
    """Read one database's labels CSV into {gene_id: {drug_class, gene_family}}.

    Values are normalized so a genuinely-missing label becomes the string
    "unknown" (not the literal text "nan" — str(float('nan')) == "nan",
    which was silently corrupting rows that had a real missing value even
    before the CD-HIT issue below).
    """
    if not labels_csv.exists():
        return {}
    df = pd.read_csv(labels_csv)
    if df.empty or "gene_id" not in df.columns:
        return {}

    lookup = {}
    for _, row in df.iterrows():
        drug_class = row.get("drug_class", "unknown")
        gene_family = row.get("gene_family", "unknown")
        lookup[row["gene_id"]] = {
            "drug_class": "unknown" if pd.isna(drug_class) else str(drug_class),
            "gene_family": "unknown" if pd.isna(gene_family) else str(gene_family),
        }
    return lookup


def load_combined_label_lookup(training_dir: Path) -> dict:
    """
    Build {"{source_db}__{gene_id}": {drug_class, gene_family}} across all
    four databases by re-reading their labels CSVs directly. This mirrors
    the "{source_db}__{original_id}" key format standardize_headers() uses,
    so it can be used to re-attach metadata by ID after CD-HIT — see
    build_final_labels() for why this is necessary.
    """
    db_labels_csvs = {
        "CARD": training_dir / "card" / "card_labels.csv",
        "AMRFinderPlus": training_dir / "amrfinderplus" / "amrfinderplus_labels.csv",
        "ResFinder": training_dir / "resfinder" / "resfinder_labels.csv",
        "MEGARes": training_dir / "megares" / "megares_labels.csv",
    }
    combined = {}
    for source_db, labels_csv in db_labels_csvs.items():
        for gene_id, meta in _label_dict_from_csv(labels_csv).items():
            combined[f"{source_db}__{gene_id}"] = meta
    return combined


@timed_step("Standardize headers")
def standardize_headers(source_fasta: Path, labels_csv: Path, source_db: str, out_fasta_handle):
    """
    Rewrite one database's FASTA with standardized headers, using its
    labels CSV (produced by download_multi_database.py) to embed gene
    symbol / drug class metadata directly in the header.
    """
    if not source_fasta.exists():
        logger.warning(f"{source_fasta} not found — skipping {source_db}")
        return 0

    label_lookup = _label_dict_from_csv(labels_csv)

    count = 0
    for record in SeqIO.parse(str(source_fasta), "fasta"):
        # gene_id in labels CSVs may already be prefixed (merged file case);
        # try both raw and prefixed forms.
        meta = label_lookup.get(record.id) or label_lookup.get(f"{source_db}__{record.id}") or {}
        # _label_dict_from_csv() already normalized missing values to the
        # string "unknown" (previously this did str(meta.get(...)), which
        # turned a genuinely-missing value into the literal text "nan"
        # instead of "unknown").
        drug_class = meta.get("drug_class", "unknown").replace(" ", "_")
        gene_family = meta.get("gene_family", "unknown").replace(" ", "_")

        new_id = f"{source_db}__{record.id}"
        record.id = new_id
        record.description = f"gene={gene_family}|class={drug_class}|source={source_db}"
        SeqIO.write(record, out_fasta_handle, "fasta")
        count += 1

    logger.info(f"{source_db}: {count} sequences standardized")
    return count


@timed_step("Merge standardized databases")
def merge_positive_sources(training_dir: Path, merged_fasta_path: Path):
    """
    training_dir is expected to contain subfolders written by
    download_multi_database.py: card/, amrfinderplus/, resfinder/, megares/
    """
    db_files = {
        "CARD": (training_dir / "card" / "card_nucleotide.fasta",
                 training_dir / "card" / "card_labels.csv"),
        "AMRFinderPlus": (training_dir / "amrfinderplus" / "amrfinderplus_nucleotide.fasta",
                           training_dir / "amrfinderplus" / "amrfinderplus_labels.csv"),
        "ResFinder": (training_dir / "resfinder" / "resfinder_nucleotide.fasta",
                      training_dir / "resfinder" / "resfinder_labels.csv"),
        "MEGARes": (training_dir / "megares" / "megares_nucleotide.fasta",
                    training_dir / "megares" / "megares_labels.csv"),
    }

    merged_fasta_path.parent.mkdir(parents=True, exist_ok=True)
    per_db_counts = {}
    with open(merged_fasta_path, "w") as out_handle:
        for db_name, (fasta_path, labels_path) in db_files.items():
            per_db_counts[db_name] = standardize_headers(fasta_path, labels_path, db_name, out_handle)

    total = sum(per_db_counts.values())
    if total == 0:
        raise DataFormatError(
            "No sequences were merged. Run scripts/download_multi_database.py "
            "first, and confirm its output landed in the expected subfolders."
        )

    logger.info(f"Merged positive set: {total} sequences (pre-dedup) -> {merged_fasta_path}")
    return per_db_counts


@timed_step("CD-HIT deduplication")
def run_cdhit(merged_fasta: Path, clustered_fasta: Path, identity: float = 0.90):
    """
    Cluster the merged positive set at the given identity threshold and
    keep one representative sequence per cluster.
    """
    cmd = [
        "cd-hit-est",
        "-i", str(merged_fasta),
        "-o", str(clustered_fasta),
        "-c", str(identity),
        "-n", "8" if identity >= 0.90 else "5",   # CD-HIT word size recommendation for this identity band
        "-M", "4000",   # memory cap (MB) — keeps this workable on 16GB RAM machines
        "-T", "0",      # use all available threads
        # -d 0: use the full ID up to the first whitespace, rather than
        # truncating it to CD-HIT's default 20-char description limit.
        # We no longer *depend* on CD-HIT preserving header text past the
        # ID (build_final_labels() re-attaches metadata by ID lookup
        # instead — see its docstring), but this is still good hygiene and
        # is the community-recommended setting for exactly this class of
        # issue: https://github.com/weizhongli/cdhit/issues/4
        "-d", "0",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(result.stdout.strip().splitlines()[-1] if result.stdout else "CD-HIT completed")
    except FileNotFoundError:
        raise ExternalToolError(
            "cd-hit-est not found on PATH. Install with: conda install -c bioconda cd-hit"
        )
    except subprocess.CalledProcessError as e:
        raise ExternalToolError(f"CD-HIT failed:\n{e.stderr}")

    return clustered_fasta


@timed_step("Build final positive labels CSV")
def build_final_labels(clustered_fasta: Path, out_csv: Path, label_lookup: dict):
    """
    Build the final labels CSV for the CD-HIT-deduplicated FASTA.

    IMPORTANT: this does NOT rely on re-parsing the "gene=...|class=...|
    source=..." text embedded in each record's description by
    standardize_headers(). CD-HIT is documented (and has open upstream
    issues, e.g. github.com/weizhongli/cdhit/issues/4) to truncate
    everything after the first whitespace in a header for at least some
    representative sequences — the embedded annotation sits right after
    that first space, so it can silently disappear for whichever records
    CD-HIT happens to truncate. AMRFinderPlus's gene_id is a long,
    pipe-delimited compound string (protein_acc|dna_acc|part|total|node|
    parent|mechanism|name), making its headers by far the longest of the
    four databases and disproportionately likely to trip this behavior —
    which is why the loss showed up concentrated in AMRFinderPlus rows.

    Instead, we use the one part of the header CD-HIT reliably preserves
    (the ID, i.e. everything before the first whitespace) to look the
    metadata back up directly from the original per-database labels CSVs
    (label_lookup, built by load_combined_label_lookup()). The embedded
    description is kept only as a last-resort fallback for any record
    whose ID isn't found in label_lookup for some other reason.
    """
    rows = []
    header_meta_pattern = re.compile(r"gene=([^|]*)\|class=([^|]*)\|source=([^|]*)")
    lookup_misses = 0

    for record in SeqIO.parse(str(clustered_fasta), "fasta"):
        source_db, original_id = (record.id.split("__", 1) + [""])[:2] if "__" in record.id else ("unknown", record.id)

        meta = label_lookup.get(record.id)
        if meta is not None:
            gene_family = meta["gene_family"]
            drug_class = meta["drug_class"]
            source = source_db
        else:
            # Fallback: try to recover it from the embedded description,
            # in case it did survive CD-HIT intact.
            match = header_meta_pattern.search(record.description)
            if match:
                gene_family, drug_class, source = match.groups()
            else:
                gene_family, drug_class, source = "unknown", "unknown", source_db
                lookup_misses += 1

        rows.append({
            "gene_id": record.id,
            "sequence_length": len(record.seq),
            "gene_family": gene_family,
            "drug_class": drug_class,
            "source_db": source,
            "label": "ARG",
        })

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Final positive dataset: {len(df)} sequences -> {out_csv}")
    logger.info(f"Sequences per source DB after dedup:\n{df['source_db'].value_counts().to_string()}")
    if lookup_misses:
        logger.warning(
            f"{lookup_misses} sequence(s) had no match in the original labels CSVs AND no "
            "recoverable embedded header metadata — these got 'unknown' for drug_class/"
            "gene_family. This means their gene_id in the CD-HIT output doesn't match "
            "'{source_db}__{original gene_id}' from download_multi_database.py; inspect a few "
            "of these gene_id values directly to see whether CD-HIT altered the ID itself."
        )
    return df


def main(training_dir: Path, outdir: Path, identity: float):
    outdir.mkdir(parents=True, exist_ok=True)

    merged_raw = outdir / "positive_merged_raw.fasta"
    per_db_downloaded = merge_positive_sources(training_dir, merged_raw)

    clustered = outdir / "positive_dedup.fasta"
    run_cdhit(merged_raw, clustered, identity=identity)

    labels_csv = outdir / "positive_labels.csv"
    label_lookup = load_combined_label_lookup(training_dir)
    final_df = build_final_labels(clustered, labels_csv, label_lookup)

    # --- Dataset version tracking (per source database) ---
    per_db_retained = final_df["source_db"].value_counts().to_dict()
    for db_name, num_downloaded in per_db_downloaded.items():
        record_dataset_version(
            source_db=db_name,
            num_downloaded=num_downloaded,
            num_retained=per_db_retained.get(db_name, 0),
            db_version=f"live download {pd.Timestamp.now().strftime('%Y-%m-%d')}",
            notes=f"CD-HIT identity={identity} applied across the merged 4-database set "
                  "(dedup is cross-database, so retained counts reflect post-merge clustering, "
                  "not per-database dedup in isolation).",
        )
    logger.info("Dataset version log updated -> reports/dataset_versions.csv")

    logger.info("Positive dataset build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the standardized, deduplicated positive ARG dataset")
    parser.add_argument("--training-dir", default="data/training",
                         help="Directory containing per-database outputs from download_multi_database.py")
    parser.add_argument("--outdir", default="data/training/positive")
    parser.add_argument("--identity", type=float, default=0.90, help="CD-HIT identity threshold")
    args = parser.parse_args()

    main(Path(args.training_dir), Path(args.outdir), args.identity)
