"""
Module 2: Gene / ORF Prediction

Wraps Prodigal (external tool) to call genes from an assembled genome
and produce nucleotide + protein FASTA outputs. Python acts as the
pipeline controller, not the gene caller.
"""
import subprocess
import shutil
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

    # Start with standard binary name
    executable = "prodigal"
    
    # Verify that the container platform can locate the binary globally
    if shutil.which(executable) is None:
        # Fallback to standard Linux binary path if PATH execution index is delayed
        if Path("/usr/bin/prodigal").exists():
            executable = "/usr/bin/prodigal"

    cmd = [
        executable,
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
            "Please ensure 'prodigal' is listed inside your packages.txt file at the repository root."
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
