"""
Module 5: DNA Sequence Feature Extraction

Computes nucleotide composition, GC content, length, and k-mer
frequency features for a DNA sequence. These form the DNA half
of the combined feature matrix (see protein_features.py for the rest).
"""
from itertools import product
from Bio import SeqIO
import pandas as pd


def nucleotide_composition(seq: str) -> dict:
    seq = seq.upper()
    length = len(seq)
    if length == 0:
        return {"A_pct": 0, "T_pct": 0, "G_pct": 0, "C_pct": 0, "GC_pct": 0, "length_bp": 0}
    a, t, g, c = seq.count("A"), seq.count("T"), seq.count("G"), seq.count("C")
    return {
        "A_pct": round(a / length * 100, 3),
        "T_pct": round(t / length * 100, 3),
        "G_pct": round(g / length * 100, 3),
        "C_pct": round(c / length * 100, 3),
        "GC_pct": round((g + c) / length * 100, 3),
        "length_bp": length,
    }


def kmer_frequencies(seq: str, k: int = 3) -> dict:
    """
    Compute normalized k-mer frequencies for a DNA sequence.
    k=3 (64 features) is a good default — increase only if the
    model underfits, since feature count grows as 4^k.
    """
    seq = seq.upper()
    bases = "ACGT"
    all_kmers = ["".join(p) for p in product(bases, repeat=k)]
    counts = {kmer: 0 for kmer in all_kmers}

    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if kmer in counts:
            counts[kmer] += 1
            total += 1

    if total == 0:
        return {f"kmer_{kmer}": 0.0 for kmer in all_kmers}

    return {f"kmer_{kmer}": round(count / total, 5) for kmer, count in counts.items()}


def extract_dna_features(seq: str, k: int = 3) -> dict:
    """Combine nucleotide composition + k-mer features for one sequence."""
    features = {}
    features.update(nucleotide_composition(seq))
    features.update(kmer_frequencies(seq, k=k))
    return features


def extract_features_from_fasta(fasta_path: str, k: int = 3) -> pd.DataFrame:
    """Build a feature matrix (one row per gene) from a multi-FASTA file."""
    rows = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        feats = extract_dna_features(str(record.seq), k=k)
        feats["gene_id"] = record.id
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("gene_id")
    return df


if __name__ == "__main__":
    import sys
    df = extract_features_from_fasta(sys.argv[1])
    print(df.head())
