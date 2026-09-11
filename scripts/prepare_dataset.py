"""
scripts/prepare_dataset.py

Final dataset preparation step, run after build_positive_dataset.py and
build_negative_set.py:

    positive_dedup.fasta + positive_labels.csv
    negative_final.fasta
        -> filter out-of-scope / too-short sequences
        -> merge into one labeled dataset
        -> cross-class redundancy check (CD-HIT across the COMBINED set)
        -> cluster-aware train / validation / test split
        -> dataset summary statistics report

Output (in data/processed/):
    train.fasta / train.csv
    val.fasta   / val.csv
    test.fasta  / test.csv

Output (in reports/):
    dataset_summary.csv   — per-split, per-class sequence counts and length stats
    dataset_summary.html  — same, human-readable

Usage:
    python scripts/prepare_dataset.py
"""
import argparse
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from Bio import SeqIO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, timed_step, DataFormatError
from src.config import cfg
from src.cluster_split import cluster_sequences, parse_clusters, cluster_aware_group_split
from src.dataset_versioning import record_dataset_version

logger = get_logger(__name__)

# Out-of-scope categories that can still appear in ARG database dumps
# (metal/biocide resistance, catch-all "multi-compound" entries). See
# the fix in download_multi_database.py's MEGARes parser for why these
# shouldn't be downloaded going forward — this filter is the safety net
# for data already on disk.
EXCLUDE_DRUG_CLASS_KEYWORDS = [
    "metal", "biocide", "copper", "mercury", "arsenic", "nickel",
    "silver", "multi-compound", "multi_compound",
]


@timed_step("Load and filter positive dataset")
def load_positive(positive_dir: Path, min_length: int) -> pd.DataFrame:
    fasta_path = positive_dir / "positive_dedup.fasta"
    labels_path = positive_dir / "positive_labels.csv"
    if not fasta_path.exists() or not labels_path.exists():
        raise DataFormatError(
            f"Expected {fasta_path} and {labels_path}. Run build_positive_dataset.py first."
        )

    labels_df = pd.read_csv(labels_path)
    seq_lengths = {r.id: len(r.seq) for r in SeqIO.parse(str(fasta_path), "fasta")}
    labels_df["sequence_length"] = labels_df["gene_id"].map(seq_lengths)
    labels_df = labels_df.dropna(subset=["sequence_length"])

    before = len(labels_df)

    # Exclude out-of-scope categories (metal/biocide resistance etc.)
    pattern = re.compile("|".join(EXCLUDE_DRUG_CLASS_KEYWORDS), re.IGNORECASE)
    is_out_of_scope = labels_df["drug_class"].astype(str).str.contains(pattern, na=False) | \
                       labels_df["gene_family"].astype(str).str.contains(pattern, na=False)
    excluded_scope = labels_df[is_out_of_scope]
    labels_df = labels_df[~is_out_of_scope]
    logger.info(f"Excluded {len(excluded_scope)} out-of-scope (metal/biocide) sequences "
                f"from the positive set: {Counter(excluded_scope['source_db']).most_common()}")

    # Exclude too-short sequences (likely Prodigal fragments / annotation errors)
    too_short = labels_df[labels_df["sequence_length"] < min_length]
    labels_df = labels_df[labels_df["sequence_length"] >= min_length]
    logger.info(f"Excluded {len(too_short)} positive sequences shorter than {min_length} bp")

    labels_df["label"] = "ARG"
    logger.info(f"Positive set after filtering: {len(labels_df)} / {before} sequences retained")
    return labels_df, fasta_path


@timed_step("Load and filter negative dataset")
def load_negative(negative_dir: Path, min_length: int) -> pd.DataFrame:
    fasta_path = negative_dir / "negative_final.fasta"
    if not fasta_path.exists():
        raise DataFormatError(f"Expected {fasta_path}. Run build_negative_set.py first.")

    rows = []
    too_short = 0
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        length = len(record.seq)
        if length < min_length:
            too_short += 1
            continue
        rows.append({
            "gene_id": record.id,
            "sequence_length": length,
            "gene_family": None,
            "drug_class": None,
            "source_db": "RefSeq_negative",
            "label": "Non-ARG",
        })

    df = pd.DataFrame(rows)
    logger.info(f"Excluded {too_short} negative sequences shorter than {min_length} bp")
    logger.info(f"Negative set after filtering: {len(df)} sequences retained")
    return df, fasta_path


@timed_step("Merge positive + negative into one FASTA")
def merge_all(pos_df, pos_fasta, neg_df, neg_fasta, combined_fasta: Path):
    combined_fasta.parent.mkdir(parents=True, exist_ok=True)
    keep_ids = set(pos_df["gene_id"]) | set(neg_df["gene_id"])

    written = 0
    with open(combined_fasta, "w") as out_handle:
        for fasta_path in (pos_fasta, neg_fasta):
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                if record.id in keep_ids:
                    SeqIO.write(record, out_handle, "fasta")
                    written += 1

    combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
    logger.info(f"Combined dataset: {written} sequences "
                f"({(combined_df['label'] == 'ARG').sum()} ARG / "
                f"{(combined_df['label'] == 'Non-ARG').sum()} Non-ARG)")
    return combined_df


@timed_step("Cross-class redundancy check (CD-HIT on combined set)")
def redundancy_check_and_cluster(combined_fasta: Path, identity: float, work_dir: Path):
    """
    Runs CD-HIT across the COMBINED positive+negative set. This serves
    two purposes at once:
      1. Sanity check — flags any negative sequence that clusters with a
         positive sequence at >=identity, which would indicate a leak
         through the negative-screening step (shouldn't happen given the
         3-database no-hit criterion, but worth confirming rather than
         assuming).
      2. Produces the cluster assignments used for the leakage-safe
         train/val/test split in the next step.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    clstr_file = cluster_sequences(str(combined_fasta), str(work_dir / "combined_clustered"),
                                    identity_threshold=identity)
    cluster_map = parse_clusters(clstr_file)
    return cluster_map


def check_cross_class_clusters(cluster_map: dict, combined_df: pd.DataFrame) -> int:
    id_to_label = dict(zip(combined_df["gene_id"], combined_df["label"]))
    clusters_to_ids = {}
    for seq_id, cluster_id in cluster_map.items():
        clusters_to_ids.setdefault(cluster_id, []).append(seq_id)

    mixed_clusters = 0
    for cluster_id, ids in clusters_to_ids.items():
        labels_in_cluster = {id_to_label.get(i) for i in ids}
        if len(labels_in_cluster) > 1:
            mixed_clusters += 1

    if mixed_clusters:
        logger.warning(f"{mixed_clusters} clusters contain BOTH ARG and Non-ARG sequences at the "
                        f"configured identity threshold — inspect these; it may indicate residual "
                        f"screening leakage. They are still kept together in the same split by the "
                        f"cluster-aware splitter below, so this does not itself cause leakage, but "
                        f"is worth reporting as a data-quality note in your methodology.")
    else:
        logger.info("No cross-class clusters found — positive and negative sets are cleanly separated.")
    return mixed_clusters


@timed_step("Cluster-aware train/val/test split")
def split_dataset(combined_df: pd.DataFrame, cluster_map: dict, train_ratio, val_ratio, test_ratio, seed):
    all_ids = combined_df["gene_id"].tolist()
    labels = dict(zip(combined_df["gene_id"], combined_df["label"]))

    assignment = cluster_aware_group_split(cluster_map, all_ids, labels,
                                            train_ratio, val_ratio, test_ratio, seed=seed)
    combined_df = combined_df.copy()
    combined_df["split"] = combined_df["gene_id"].map(assignment)
    return combined_df


def write_split_files(combined_df: pd.DataFrame, combined_fasta: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    id_to_record = {r.id: r for r in SeqIO.parse(str(combined_fasta), "fasta")}

    for split_name in ["train", "val", "test"]:
        split_df = combined_df[combined_df["split"] == split_name]
        split_df.to_csv(outdir / f"{split_name}.csv", index=False)

        fasta_out = outdir / f"{split_name}.fasta"
        with open(fasta_out, "w") as f:
            for gene_id in split_df["gene_id"]:
                record = id_to_record.get(gene_id)
                if record:
                    SeqIO.write(record, f, "fasta")

        logger.info(f"{split_name}: {len(split_df)} sequences "
                    f"({(split_df['label'] == 'ARG').sum()} ARG / "
                    f"{(split_df['label'] == 'Non-ARG').sum()} Non-ARG) -> {fasta_out}")


@timed_step("Write dataset summary report")
def write_summary_report(combined_df: pd.DataFrame, mixed_clusters: int, reports_dir: Path):
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for split_name in ["train", "val", "test"]:
        split_df = combined_df[combined_df["split"] == split_name]
        for label in ["ARG", "Non-ARG"]:
            subset = split_df[split_df["label"] == label]
            summary_rows.append({
                "split": split_name,
                "label": label,
                "count": len(subset),
                "mean_length_bp": round(subset["sequence_length"].mean(), 1) if len(subset) else 0,
                "min_length_bp": int(subset["sequence_length"].min()) if len(subset) else 0,
                "max_length_bp": int(subset["sequence_length"].max()) if len(subset) else 0,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = reports_dir / "dataset_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    html_path = reports_dir / "dataset_summary.html"
    html = f"""
    <html><head><title>Dataset Summary</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
        th {{ background: #1F3864; color: white; }}
    </style></head><body>
    <h1>AMR-PREDICT Dataset Summary</h1>
    <p>Cross-class mixed clusters detected: {mixed_clusters}</p>
    {summary_df.to_html(index=False)}
    </body></html>
    """
    html_path.write_text(html)
    logger.info(f"Dataset summary -> {summary_csv}, {html_path}")
    return summary_df


def main(positive_dir: Path, negative_dir: Path, processed_dir: Path, reports_dir: Path):
    identity = cfg("dataset.cdhit_identity", 0.90)
    min_length = cfg("dataset.min_gene_length_bp", 100)
    train_ratio = cfg("dataset.train_ratio", 0.70)
    val_ratio = cfg("dataset.val_ratio", 0.15)
    test_ratio = cfg("dataset.test_ratio", 0.15)
    seed = cfg("random_seed", 42)

    pos_df, pos_fasta = load_positive(positive_dir, min_length)
    neg_df, neg_fasta = load_negative(negative_dir, min_length)

    combined_fasta = processed_dir / "combined_dataset.fasta"
    combined_df = merge_all(pos_df, pos_fasta, neg_df, neg_fasta, combined_fasta)

    cluster_map = redundancy_check_and_cluster(combined_fasta, identity, processed_dir / "_cdhit_work")
    mixed_clusters = check_cross_class_clusters(cluster_map, combined_df)

    combined_df = split_dataset(combined_df, cluster_map, train_ratio, val_ratio, test_ratio, seed)
    write_split_files(combined_df, combined_fasta, processed_dir)
    write_summary_report(combined_df, mixed_clusters, reports_dir)

    record_dataset_version(
        source_db="Combined (positive+negative, post-filter)",
        num_downloaded=len(pos_df) + len(neg_df),
        num_retained=len(combined_df),
        db_version=f"prepared {pd.Timestamp.now().strftime('%Y-%m-%d')}, "
                    f"CD-HIT identity={identity}, min_length={min_length}bp",
        notes=f"train/val/test ratios={train_ratio}/{val_ratio}/{test_ratio}, seed={seed}, "
              f"cross-class mixed clusters={mixed_clusters}",
    )

    logger.info("Dataset preparation complete. Proceed to feature extraction next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge, filter, dedup-check, and split the final ARG dataset")
    parser.add_argument("--positive-dir", default="data/training/positive")
    parser.add_argument("--negative-dir", default="data/training/negative")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    main(Path(args.positive_dir), Path(args.negative_dir), Path(args.processed_dir), Path(args.reports_dir))
