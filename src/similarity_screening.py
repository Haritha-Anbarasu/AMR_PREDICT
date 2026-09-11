"""
Module 3/4: Similarity-Based ARG Screening

Wraps ABRicate (simplest option) or RGI/CARD to screen predicted genes
against a curated ARG reference database.

Install ABRicate first:
    conda install -c bioconda -c conda-forge abricate
    abricate --setupdb

Usage note: ABRicate expects nucleotide gene sequences (the *_genes.fna
output from gene_prediction.py), one FASTA of candidate genes.
"""
import subprocess
import csv
from pathlib import Path


def run_arg_screening(genes_fasta: str, output_csv: str, db: str = "card") -> str:
    """
    Run ABRicate against the given ARG database (default: CARD).
    Writes a tab-separated results file and returns its path.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["abricate", "--db", db, str(genes_fasta)]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ABRicate is not installed or not on PATH. "
            "Install it with: conda install -c bioconda -c conda-forge abricate"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ABRicate failed:\n{e.stderr}")

    output_csv.write_text(result.stdout)
    return str(output_csv)


def parse_screening_results(abricate_output: str) -> list:
    """
    Parse ABRicate's tab-separated output into a list of dicts with
    gene, identity, coverage, resistance_class, and a simple
    similarity call (High/Moderate/Low) based on %identity.
    """
    hits = []
    with open(abricate_output) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            identity = float(row.get("%IDENTITY", 0))
            coverage = float(row.get("%COVERAGE", 0))

            if identity >= 90:
                similarity = "High"
            elif identity >= 60:
                similarity = "Moderate"
            else:
                similarity = "Low"

            hits.append({
                "gene": row.get("SEQUENCE"),
                "identity_pct": identity,
                "coverage_pct": coverage,
                "similarity": similarity,
                "resistance_gene": row.get("GENE"),
                "resistance_class": row.get("RESISTANCE", "Unknown"),
                "result": "Known ARG" if similarity == "High" else
                          "Candidate ARG" if similarity == "Moderate" else "Unknown",
            })
    return hits


if __name__ == "__main__":
    import sys
    out = run_arg_screening(sys.argv[1], sys.argv[2])
    print(parse_screening_results(out))
