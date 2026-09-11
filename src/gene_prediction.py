"""
Module 2: Gene / ORF Prediction

Wraps Prodigal (external tool) to call genes from an assembled genome
and produce nucleotide + protein FASTA outputs. Python acts as the
pipeline controller, not the gene caller.

Install Prodigal first:
    conda install -c bioconda prodigal
    (or) sudo apt-get install prodigal
"""
import subprocess
from pathlib import Path


def predict_genes(genome_fasta: str, output_dir: str, prefix: str = "genome") -> dict:
    """
    Run Prodigal on an input genome FASTA.

    Returns dict with paths to:
        genes_nt_fasta, genes_aa_fasta, gff
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nt_out = output_dir / f"{prefix}_genes.fna"
    aa_out = output_dir / f"{prefix}_genes.faa"
    gff_out = output_dir / f"{prefix}_genes.gff"

    cmd = [
        "prodigal",
        "-i", str(genome_fasta),
        "-d", str(nt_out),   # nucleotide gene sequences
        "-a", str(aa_out),   # protein translations
        "-f", "gff",
        "-o", str(gff_out),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "Prodigal is not installed or not on PATH. "
            "Install it with: conda install -c bioconda prodigal"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Prodigal failed:\n{e.stderr}")

    return {
        "genes_nt_fasta": str(nt_out),
        "genes_aa_fasta": str(aa_out),
        "gff": str(gff_out),
    }


if __name__ == "__main__":
    import sys
    result = predict_genes(sys.argv[1], sys.argv[2])
    print(result)
