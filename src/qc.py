"""
Module 1: Genome Quality Control
Computes basic QC metrics for an input bacterial genome (FASTA).
"""
from pathlib import Path
from Bio import SeqIO


def compute_genome_qc(fasta_path: str) -> dict:
    """
    Compute QC metrics for a genome FASTA file.

    Returns a dict with:
        genome_size_bp, gc_content_pct, n_content_pct,
        num_contigs, quality_status
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"Genome file not found: {fasta_path}")

    total_len = 0
    gc_count = 0
    n_count = 0
    num_contigs = 0

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq = str(record.seq).upper()
        total_len += len(seq)
        gc_count += seq.count("G") + seq.count("C")
        n_count += seq.count("N")
        num_contigs += 1

    if total_len == 0:
        raise ValueError("No sequences found in FASTA file.")

    gc_pct = round((gc_count / total_len) * 100, 2)
    n_pct = round((n_count / total_len) * 100, 2)

    # Simple quality heuristic — tune thresholds as needed during validation
    quality_status = "Acceptable" if n_pct < 1.0 and total_len > 100_000 else "Review Needed"

    return {
        "genome_size_bp": total_len,
        "genome_size_mb": round(total_len / 1_000_000, 2),
        "gc_content_pct": gc_pct,
        "n_content_pct": n_pct,
        "num_contigs": num_contigs,
        "quality_status": quality_status,
    }


if __name__ == "__main__":
    import sys
    result = compute_genome_qc(sys.argv[1])
    for k, v in result.items():
        print(f"{k}: {v}")
