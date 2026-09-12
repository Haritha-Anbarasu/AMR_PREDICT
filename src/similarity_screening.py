"""
Module 3/4: Similarity-Based ARG Screening (BLAST+ implementation)

Drop-in replacement for the ABRicate-based similarity_screening.py.
ABRicate itself is just a wrapper around BLAST+ with bundled reference
databases, and it isn't installable on Streamlit Community Cloud (no
conda support there). This module does the same job directly:

    1. Build (once, cached) a nucleotide BLAST database from CARD's
       "protein homolog model" FASTA — the same reference set ABRicate's
       CARD db is built from. That FASTA already ships in this repo at
       data/raw_downloads/card/, so no runtime download is needed.
    2. blastn the predicted genes against that database.
    3. Take the best hit per gene (highest bitscore).
    4. Look up each hit's drug class from CARD's aro_index.tsv, keyed by
       the ARO accession embedded in the CARD FASTA header.
    5. Apply the same %identity thresholds ABRicate used (>=90% High,
       >=60% Moderate, else Low) so the rest of the pipeline
       (decision_engine, report_generation) needs no changes.

Requirements:
    - BLAST+ on PATH (`blastn`, `makeblastdb`) — apt package `ncbi-blast+`,
      already available on Streamlit Community Cloud via packages.txt.
    - CARD reference FASTA + aro_index.tsv bundled under
      data/raw_downloads/card/ (already present in this repo).

Output format matches the original ABRicate-based module's CSV/TSV
columns (SEQUENCE, %IDENTITY, %COVERAGE, GENE, RESISTANCE), so
parse_screening_results() and everything downstream is unchanged.
"""
import csv
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARD_DIR = PROJECT_ROOT / "data" / "raw_downloads" / "card"
CARD_NT_FASTA = CARD_DIR / "nucleotide_fasta_protein_homolog_model.fasta"
CARD_ARO_INDEX = CARD_DIR / "aro_index.tsv"
BLAST_DB_DIR = PROJECT_ROOT / "data" / "blastdb"
BLAST_DB_PATH = BLAST_DB_DIR / "card_nt"


def _ensure_blast_available():
    for tool in ("blastn", "makeblastdb"):
        try:
            subprocess.run([tool, "-version"], capture_output=True, check=True)
        except FileNotFoundError:
            raise RuntimeError(
                f"{tool} is not installed or not on PATH. "
                "Install BLAST+ with: apt-get install ncbi-blast+ "
                "(already listed in packages.txt for Streamlit Community Cloud)."
            )


def _build_blast_db_if_needed(fasta: Path = CARD_NT_FASTA, db_path: Path = BLAST_DB_PATH) -> Path:
    """
    Build the nucleotide BLAST database from the CARD reference FASTA the
    first time it's needed, then reuse it on every later call/run.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    marker = db_path.with_suffix(".nsq")  # one of the files makeblastdb produces

    if marker.exists():
        return db_path

    if not fasta.exists():
        raise RuntimeError(
            f"CARD reference FASTA not found at {fasta}. "
            "This should ship with the repo under data/raw_downloads/card/."
        )

    cmd = ["makeblastdb", "-in", str(fasta), "-dbtype", "nucl", "-out", str(db_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"makeblastdb failed:\n{e.stderr}")

    return db_path


def _load_aro_drug_classes(aro_index: Path = CARD_ARO_INDEX) -> dict:
    """Map ARO accession (e.g. 'ARO:3002999') -> Drug Class string from CARD's aro_index.tsv."""
    mapping = {}
    if not aro_index.exists():
        return mapping
    with open(aro_index, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            aro = row.get("ARO Accession", "").strip()
            drug_class = row.get("Drug Class", "").strip()
            if aro:
                mapping[aro] = drug_class or "Unknown"
    return mapping


def _parse_card_header(sseqid: str) -> tuple:
    """
    CARD nucleotide FASTA headers look like:
        gb|GQ343019.1|+|132-1023|ARO:3002999|CblA-1
    Returns (aro_accession, gene_name).
    """
    parts = sseqid.split("|")
    aro_accession = next((p for p in parts if p.startswith("ARO:")), "")
    gene_name = parts[-1] if parts else sseqid
    return aro_accession, gene_name


def run_arg_screening(genes_fasta: str, output_csv: str, db: str = "card") -> str:
    """
    BLAST the given nucleotide gene FASTA against the CARD reference set
    and write a tab-separated results file with the same column names the
    original ABRicate-based module produced (SEQUENCE, %IDENTITY,
    %COVERAGE, GENE, RESISTANCE). Returns the output path.
    """
    if db != "card":
        raise NotImplementedError(
            "This BLAST+ implementation currently ships only the CARD "
            "reference database. Add another bundled FASTA + index to "
            "extend it to other databases."
        )

    _ensure_blast_available()
    db_path = _build_blast_db_if_needed()
    drug_classes = _load_aro_drug_classes()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore"]
    cmd = [
        "blastn",
        "-query", str(genes_fasta),
        "-db", str(db_path),
        "-outfmt", "6 " + " ".join(fields),
        "-max_target_seqs", "5",
        "-evalue", "1e-10",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"blastn failed:\n{e.stderr}")

    # Keep only the best hit (highest bitscore) per query gene.
    best_hits = {}
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        row = dict(zip(fields, line.split("\t")))
        qseqid = row["qseqid"]
        bitscore = float(row["bitscore"])
        if qseqid not in best_hits or bitscore > float(best_hits[qseqid]["bitscore"]):
            best_hits[qseqid] = row

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["SEQUENCE", "%IDENTITY", "%COVERAGE", "GENE", "RESISTANCE"])
        for qseqid, row in best_hits.items():
            identity = float(row["pident"])
            coverage = round(float(row["length"]) / float(row["qlen"]) * 100, 2)
            aro_accession, gene_name = _parse_card_header(row["sseqid"])
            resistance = drug_classes.get(aro_accession, "Unknown")
            writer.writerow([qseqid, identity, coverage, gene_name, resistance])

    return str(output_csv)


def parse_screening_results(blast_output: str) -> list:
    """
    Parse the TSV written by run_arg_screening() into a list of dicts with
    gene, identity, coverage, resistance_class, and a simple similarity
    call (High/Moderate/Low) based on %identity. Same thresholds and
    output shape as the original ABRicate-based module.
    """
    hits = []
    with open(blast_output) as f:
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
