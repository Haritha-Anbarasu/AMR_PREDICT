"""
download_card_data.py

Automatically downloads the CARD (Comprehensive Antibiotic Resistance
Database) data bundle, extracts it, and builds a labeled training set
(FASTA + CSV) with AMR Gene Family / Drug Class / Resistance Mechanism
annotations for every sequence — ready to feed into
src/sequence_features.py and src/protein_features.py.

Requires internet access (this reaches out to card.mcmaster.ca, which
is NOT reachable from a sandboxed/offline environment — run this on
your own machine).

Usage:
    python scripts/download_card_data.py --outdir data/training

Output files (in --outdir):
    card_nucleotide.fasta   - nucleotide sequences for ARG reference genes
    card_protein.fasta      - protein sequences for the same genes
    card_labels.csv         - one row per gene: aro_accession, aro_name,
                               drug_class, resistance_mechanism, amr_gene_family
"""
import argparse
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

CARD_DATA_URL = "https://card.mcmaster.ca/latest/data"

# CARD's data bundle includes several model-type FASTA files. We use the
# "protein homolog model" ones — these are the pure similarity-detectable
# ARGs, which pairs naturally with ABRicate/RGI screening in the rest of
# this pipeline. (Other model types — variant, rRNA, knockout, overexpression
# models — capture resistance via point mutations, not gene presence, and
# need a different detection strategy.)
NUCLEOTIDE_FASTA_NAME = "nucleotide_fasta_protein_homolog_model.fasta"
PROTEIN_FASTA_NAME = "protein_fasta_protein_homolog_model.fasta"
ARO_INDEX_NAME = "aro_index.tsv"

ARO_PATTERN = re.compile(r"ARO:(\d+)")


def download_card_bundle(dest_dir: Path) -> Path:
    """Download and extract the CARD data tarball."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = dest_dir / "card-data.tar.bz2"

    print(f"Downloading CARD data bundle from {CARD_DATA_URL} ...")
    urllib.request.urlretrieve(CARD_DATA_URL, tarball_path)

    print("Extracting ...")
    with tarfile.open(tarball_path, "r:bz2") as tar:
        tar.extractall(dest_dir)

    return dest_dir


def load_aro_index(card_dir: Path) -> pd.DataFrame:
    """
    Load aro_index.tsv, which maps ARO Accession -> AMR Gene Family,
    Drug Class, Resistance Mechanism (the labels we need for the
    multiclass model).
    """
    aro_index_path = card_dir / ARO_INDEX_NAME
    if not aro_index_path.exists():
        raise FileNotFoundError(
            f"{ARO_INDEX_NAME} not found in {card_dir}. "
            "The CARD bundle format may have changed — check the extracted "
            "files with `ls` and update ARO_INDEX_NAME accordingly."
        )
    df = pd.read_csv(aro_index_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df


def extract_aro_id(header: str) -> str:
    """Pull the numeric ARO accession out of a CARD FASTA header."""
    match = ARO_PATTERN.search(header)
    return match.group(1) if match else None


def parse_fasta_headers(fasta_path: Path) -> pd.DataFrame:
    """Return a DataFrame of {header, aro_id, sequence_id} for a CARD FASTA file."""
    rows = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                rows.append({
                    "header": header,
                    "sequence_id": header.split(" ")[0].split("|")[0],
                    "aro_id": extract_aro_id(header),
                })
    return pd.DataFrame(rows)


def build_labeled_dataset(card_dir: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    nt_fasta = card_dir / NUCLEOTIDE_FASTA_NAME
    aa_fasta = card_dir / PROTEIN_FASTA_NAME

    if not nt_fasta.exists() or not aa_fasta.exists():
        raise FileNotFoundError(
            f"Expected FASTA files not found in {card_dir}. Run `ls {card_dir}` "
            "to see what was actually extracted and update the file names at "
            "the top of this script if CARD has renamed them."
        )

    print("Parsing ARO index (labels) ...")
    aro_df = load_aro_index(card_dir)

    # aro_index.tsv columns are typically:
    # 'ARO Accession', 'AMR Gene Family', 'Drug Class', 'Resistance Mechanism', ...
    aro_df["aro_id"] = aro_df["ARO Accession"].str.extract(r"ARO:(\d+)")

    print("Parsing nucleotide FASTA headers ...")
    nt_headers = parse_fasta_headers(nt_fasta)

    print("Merging sequences with labels ...")
    merged = nt_headers.merge(aro_df, on="aro_id", how="left")

    unmatched = merged["Drug Class"].isna().sum() if "Drug Class" in merged.columns else len(merged)
    if unmatched:
        print(f"Warning: {unmatched} of {len(merged)} sequences could not be matched to a label. "
              "These rows will still appear in the CSV with blank labels — inspect and drop as needed.")

    label_cols = [c for c in ["ARO Accession", "AMR Gene Family", "Drug Class", "Resistance Mechanism"]
                  if c in merged.columns]
    labels_csv = merged[["sequence_id", "aro_id"] + label_cols].rename(columns={
        "ARO Accession": "aro_accession",
        "AMR Gene Family": "amr_gene_family",
        "Drug Class": "drug_class",
        "Resistance Mechanism": "resistance_mechanism",
    })

    labels_out = outdir / "card_labels.csv"
    labels_csv.to_csv(labels_out, index=False)

    # Copy the sequence FASTAs alongside the labels for convenience
    nt_out = outdir / "card_nucleotide.fasta"
    aa_out = outdir / "card_protein.fasta"
    nt_out.write_bytes(nt_fasta.read_bytes())
    aa_out.write_bytes(aa_fasta.read_bytes())

    print(f"\nDone.")
    print(f"  Labels:     {labels_out}  ({len(labels_csv)} sequences)")
    print(f"  Nucleotide: {nt_out}")
    print(f"  Protein:    {aa_out}")

    if "drug_class" in labels_csv.columns:
        print("\nDrug class distribution (top 15):")
        print(labels_csv["drug_class"].value_counts().head(15).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare CARD training data")
    parser.add_argument("--outdir", default="data/training", help="Where to write the labeled dataset")
    parser.add_argument("--raw-dir", default="data/raw_card", help="Where to download/extract the raw CARD bundle")
    parser.add_argument("--skip-download", action="store_true",
                         help="Skip downloading, use an already-extracted bundle in --raw-dir")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    outdir = Path(args.outdir)

    if not args.skip_download:
        download_card_bundle(raw_dir)
    else:
        print(f"Skipping download, using existing files in {raw_dir}")

    build_labeled_dataset(raw_dir, outdir)
