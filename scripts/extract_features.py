"""
scripts/extract_features.py

Builds the final numeric feature matrix for each of train/val/test:

    data/processed/{split}.fasta + {split}.csv (from prepare_dataset.py)
        -> DNA features (composition, k-mer, codon usage bias)
        -> translate to protein (NCBI table 11 — bacterial code, not
           the eukaryotic default — see translate_bacterial())
        -> protein features (composition, MW, pI, GRAVY, aromaticity,
           instability index, aliphatic index)
        -> merged feature matrix + label, written to
           data/processed/features/{split}_features.csv

Why translation table 11 specifically: Biopython's Seq.translate()
defaults to the standard eukaryotic genetic code (NCBI table 1). Every
sequence in this pipeline comes from a bacterial genome or a
bacterial-focused ARG database, and bacteria (along with archaea and
plant plastids) use NCBI table 11, which differs from table 1 in stop
codon usage and a small number of start codon assignments. Using the
wrong table would occasionally produce spurious internal stop codons
or mistranslated residues — small in isolation, but wrong for a
dissertation to gloss over silently.

Usage:
    python scripts/extract_features.py
"""
import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, timed_step, get_progress_bar, DataFormatError
from src.config import cfg
from src.sequence_features import extract_dna_features
from src.protein_features import extract_protein_features

logger = get_logger(__name__)

BACTERIAL_TRANSLATION_TABLE = 11


def translate_bacterial(nt_seq: str) -> str:
    """
    Translate a nucleotide gene sequence to protein using NCBI table 11
    (bacterial/archaeal/plant plastid code), trimming any trailing
    1-2 incomplete bases and stripping a single trailing stop codon
    if present (translate(to_stop=True) handles internal-vs-trailing
    stops correctly for a well-formed ORF).
    """
    seq = nt_seq.upper().replace("N", "")  # drop ambiguous bases rather than guess
    trim = len(seq) - (len(seq) % 3)
    seq = seq[:trim]
    if len(seq) == 0:
        return ""
    try:
        protein = str(Seq(seq).translate(table=BACTERIAL_TRANSLATION_TABLE, to_stop=True))
    except Exception:
        # Malformed sequence (e.g. non-ACGT characters slipping through) — skip gracefully
        return ""
    return protein


@timed_step("Extract features for one split")
def extract_split_features(fasta_path: Path, labels_csv: Path, kmer_size: int) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_csv).set_index("gene_id")

    rows = []
    skipped = []
    records = list(SeqIO.parse(str(fasta_path), "fasta"))

    for record in get_progress_bar(records, desc=f"Extracting features ({fasta_path.stem})"):
        gene_id = record.id
        nt_seq = str(record.seq)

        try:
            dna_feats = extract_dna_features(nt_seq, k=kmer_size)
        except Exception as e:
            logger.warning(f"DNA feature extraction failed for {gene_id}: {e}")
            skipped.append(gene_id)
            continue

        protein_seq = translate_bacterial(nt_seq)
        if not protein_seq:
            logger.warning(f"Translation produced an empty protein for {gene_id} — "
                            f"skipping (sequence length {len(nt_seq)}bp, possibly malformed)")
            skipped.append(gene_id)
            continue

        try:
            protein_feats = extract_protein_features(protein_seq)
        except Exception as e:
            logger.warning(f"Protein feature extraction failed for {gene_id}: {e}")
            skipped.append(gene_id)
            continue

        combined = {**dna_feats, **protein_feats}
        combined["gene_id"] = gene_id

        if gene_id in labels_df.index:
            combined["label"] = labels_df.loc[gene_id, "label"]
            combined["drug_class"] = labels_df.loc[gene_id].get("drug_class")
            combined["source_db"] = labels_df.loc[gene_id].get("source_db")
        else:
            logger.warning(f"{gene_id} not found in {labels_csv.name} — label will be missing")
            combined["label"] = None

        rows.append(combined)

    if skipped:
        logger.warning(f"{fasta_path.stem}: skipped {len(skipped)} / {len(records)} sequences "
                        f"during feature extraction (see warnings above for reasons)")

    df = pd.DataFrame(rows).set_index("gene_id")
    logger.info(f"{fasta_path.stem}: extracted {len(df)} feature rows, {df.shape[1]} feature columns")
    return df


def main(processed_dir: Path, out_dir: Path):
    kmer_size = cfg("features.kmer_size", 3)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for split_name in ["train", "val", "test"]:
        fasta_path = processed_dir / f"{split_name}.fasta"
        csv_path = processed_dir / f"{split_name}.csv"
        if not fasta_path.exists() or not csv_path.exists():
            raise DataFormatError(
                f"Expected {fasta_path} and {csv_path}. Run scripts/prepare_dataset.py first."
            )

        df = extract_split_features(fasta_path, csv_path, kmer_size)
        out_path = out_dir / f"{split_name}_features.csv"
        df.to_csv(out_path)
        logger.info(f"{split_name} features -> {out_path}")

        summary_rows.append({
            "split": split_name,
            "n_sequences": len(df),
            "n_features": df.shape[1] - 3,  # exclude label/drug_class/source_db from the count
            "n_ARG": (df["label"] == "ARG").sum(),
            "n_Non_ARG": (df["label"] == "Non-ARG").sum(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(cfg("paths.reports_dir", "reports")) / "feature_extraction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Feature extraction summary -> {summary_path}\n{summary_df.to_string(index=False)}")
    logger.info("Feature extraction complete. Proceed to model training next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DNA + protein features for train/val/test splits")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed/features")
    args = parser.parse_args()

    main(Path(args.processed_dir), Path(args.out_dir))
