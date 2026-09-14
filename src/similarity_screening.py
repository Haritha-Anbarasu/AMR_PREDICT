"""
Module 3/4: Similarity-Based ARG Screening (BLAST+ implementation, multi-database)

BLAST+-based replacement for the ABRicate-based similarity_screening.py.
Screens predicted genes against FOUR reference databases instead of just
CARD:

    - CARD          (data/raw_downloads/card/)
    - ResFinder      (data/raw_downloads/resfinder/resfinder_db/)
    - MEGARes        (data/raw_downloads/megares/)
    - AMRFinderPlus  (data/raw_downloads/amrfinderplus/)

For each database a separate nucleotide BLAST database is built (once,
cached under data/blastdb/). The predicted genes are BLASTed against all
four independently. For each query gene, the single best-scoring hit
(highest bitscore) across ALL FOUR databases combined is kept, so the
output has exactly one row per gene — same as the original CARD-only
module — with an added DATABASE column recording which reference the
winning hit came from.

Database-specific notes:
    - CARD: drug class looked up from aro_index.tsv via the ARO accession
      embedded in the FASTA header.
    - ResFinder: drug class looked up from phenotypes.txt via an exact
      match on the FASTA header (ResFinder headers are unique gene IDs).
    - MEGARes: type/class/gene name are embedded directly in the header
      pipe-delimited fields. Only header type "Drugs" is kept (Biocides /
      Metals / Multi-compound resistance entries are excluded — this
      module is scoped to antibiotic resistance).
    - AMRFinderPlus: AMR_CDS.fa also contains virulence and stress-response
      genes alongside true AMR genes. Hits are cross-referenced against
      ReferenceGeneCatalog.txt by protein accession and only kept if
      catalogued type == "AMR".

Output columns (TSV): SEQUENCE, %IDENTITY, %COVERAGE, GENE, RESISTANCE,
DATABASE. parse_screening_results() reads GENE/RESISTANCE/etc. by name,
so the extra DATABASE column doesn't break anything downstream.
"""
import csv
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_downloads"
BLAST_DB_DIR = PROJECT_ROOT / "data" / "blastdb"

CARD_DIR = RAW_DIR / "card"
RESFINDER_DIR = RAW_DIR / "resfinder" / "resfinder_db"
MEGARES_DIR = RAW_DIR / "megares"
AMRFINDER_DIR = RAW_DIR / "amrfinderplus"

BLAST_OUTFMT_FIELDS = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore"]


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


def _build_blast_db_if_needed(fasta: Path, db_path: Path) -> Path:
    """Build a nucleotide BLAST database from `fasta` the first time it's
    needed, then reuse it on every later call/run."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    marker = db_path.with_suffix(".nsq")

    if marker.exists():
        return db_path

    if not fasta.exists():
        raise RuntimeError(
            f"Reference FASTA not found at {fasta}. "
            "This should ship with the repo under data/raw_downloads/."
        )

    cmd = ["makeblastdb", "-in", str(fasta), "-dbtype", "nucl", "-out", str(db_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"makeblastdb failed for {fasta.name}:\n{e.stderr}")

    return db_path


# ---------------------------------------------------------------------------
# CARD
# ---------------------------------------------------------------------------

def _load_card_drug_classes() -> dict:
    aro_index = CARD_DIR / "aro_index.tsv"
    mapping = {}
    if not aro_index.exists():
        return mapping
    with open(aro_index, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            aro = row.get("ARO Accession", "").strip()
            drug_class = row.get("Drug Class", "").strip()
            if aro:
                mapping[aro] = drug_class or "Unknown"
    return mapping


def _parse_card_hit(sseqid: str, drug_classes: dict):
    """CARD headers: gb|GQ343019.1|+|132-1023|ARO:3002999|CblA-1"""
    parts = sseqid.split("|")
    aro_accession = next((p for p in parts if p.startswith("ARO:")), "")
    gene_name = parts[-1] if parts else sseqid
    resistance = drug_classes.get(aro_accession, "Unknown")
    return gene_name, resistance


# ---------------------------------------------------------------------------
# ResFinder
# ---------------------------------------------------------------------------

def _load_resfinder_phenotypes() -> dict:
    """Map full ResFinder header (e.g. "aac(6')-Ib_2_M23634") -> Class."""
    phenotypes = RESFINDER_DIR / "phenotypes.txt"
    mapping = {}
    if not phenotypes.exists():
        return mapping
    with open(phenotypes, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene_id = (row.get("Gene_accession no.") or "").strip()
            drug_class = (row.get("Class") or "").strip()
            if gene_id:
                mapping[gene_id] = drug_class or "Unknown"
    return mapping


def _parse_resfinder_hit(sseqid: str, phenotypes: dict):
    """ResFinder headers ARE the gene id, e.g. "aac(6')-Ib_2_M23634"."""
    gene_name = sseqid.split()[0]
    resistance = phenotypes.get(gene_name, "Unknown")
    return gene_name, resistance


# ---------------------------------------------------------------------------
# MEGARes
# ---------------------------------------------------------------------------

def _parse_megares_hit(sseqid: str):
    """
    MEGARes v3 headers:
        MEG_1|Drugs|Aminoglycosides|Aminoglycoside-resistant_16S...|A16S|RequiresSNPConfirmation
    Fields: id | type | class | mechanism | gene | snp_flag
    Only "Drugs" type entries are kept (antibiotic resistance, not
    biocide/metal resistance). Returns None if this hit should be excluded.
    """
    parts = sseqid.split("|")
    if len(parts) < 5:
        return None
    header_type, drug_class, _mechanism, gene_name = parts[1], parts[2], parts[3], parts[4]
    if header_type != "Drugs":
        return None
    return gene_name, drug_class


# ---------------------------------------------------------------------------
# AMRFinderPlus
# ---------------------------------------------------------------------------

def _load_amrfinder_catalog() -> dict:
    """Map protein accession -> (type, class, subclass, gene_family)."""
    catalog = AMRFINDER_DIR / "ReferenceGeneCatalog.txt"
    mapping = {}
    if not catalog.exists():
        return mapping
    with open(catalog, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene_type = (row.get("type") or "").strip()
            drug_class = (row.get("class") or "").strip()
            subclass = (row.get("subclass") or "").strip()
            gene_family = (row.get("gene_family") or "").strip()
            for acc_col in ("refseq_protein_accession", "genbank_protein_accession"):
                acc = (row.get(acc_col) or "").strip()
                if acc:
                    mapping[acc] = (gene_type, drug_class, subclass, gene_family)
    return mapping


def _parse_amrfinder_hit(sseqid: str, catalog: dict):
    """
    AMR_CDS.fa headers:
        WP_000027057.1|NG_050145.1|1|1|blaTEM-1|blaTEM|broad-spectrum...
    First field is the protein accession, cross-referenced against
    ReferenceGeneCatalog.txt. AMR_CDS.fa also contains VIRULENCE and
    STRESS genes; only type == "AMR" is kept. Returns None to exclude.
    """
    parts = sseqid.split("|")
    accession = parts[0] if parts else sseqid
    entry = catalog.get(accession)
    if entry is None:
        return None
    gene_type, drug_class, subclass, gene_family = entry
    if gene_type != "AMR":
        return None
    gene_name = gene_family or (parts[4] if len(parts) > 4 else accession)
    resistance = drug_class or subclass or "Unknown"
    return gene_name, resistance


# ---------------------------------------------------------------------------
# Database registry
# ---------------------------------------------------------------------------

def _get_databases():
    """
    Returns a list of dicts, one per reference database, each with:
      name, fasta path, blast db path, and a parse_fn(sseqid) -> (gene, resistance) or None
    Loaded lazily so metadata files are only read once per run.
    """
    card_classes = _load_card_drug_classes()
    resfinder_phenotypes = _load_resfinder_phenotypes()
    amrfinder_catalog = _load_amrfinder_catalog()

    return [
        {
            "name": "CARD",
            "fasta": CARD_DIR / "nucleotide_fasta_protein_homolog_model.fasta",
            "db_path": BLAST_DB_DIR / "card_nt",
            "parse_fn": lambda sseqid: _parse_card_hit(sseqid, card_classes),
        },
        {
            "name": "ResFinder",
            "fasta": RESFINDER_DIR / "all.fsa",
            "db_path": BLAST_DB_DIR / "resfinder_nt",
            "parse_fn": lambda sseqid: _parse_resfinder_hit(sseqid, resfinder_phenotypes),
        },
        {
            "name": "MEGARes",
            "fasta": MEGARES_DIR / "megares_database_v3.00.fasta",
            "db_path": BLAST_DB_DIR / "megares_nt",
            "parse_fn": _parse_megares_hit,
        },
        {
            "name": "AMRFinderPlus",
            "fasta": AMRFINDER_DIR / "AMR_CDS.fa",
            "db_path": BLAST_DB_DIR / "amrfinderplus_nt",
            "parse_fn": lambda sseqid: _parse_amrfinder_hit(sseqid, amrfinder_catalog),
        },
    ]


def _blastn(genes_fasta: str, db_path: Path) -> list:
    cmd = [
        "blastn",
        "-query", str(genes_fasta),
        "-db", str(db_path),
        "-outfmt", "6 " + " ".join(BLAST_OUTFMT_FIELDS),
        "-max_target_seqs", "5",
        "-evalue", "1e-10",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"blastn failed against {db_path.name}:\n{e.stderr}")

    rows = []
    for line in result.stdout.strip().splitlines():
        if line:
            rows.append(dict(zip(BLAST_OUTFMT_FIELDS, line.split("\t"))))
    return rows


def run_arg_screening(genes_fasta: str, output_csv: str, db: str = "all") -> str:
    """
    BLAST the given nucleotide gene FASTA against CARD, ResFinder, MEGARes,
    and AMRFinderPlus. For each query gene, keep the single best-scoring
    hit (highest bitscore) across all four databases combined. Writes a
    TSV with columns SEQUENCE, %IDENTITY, %COVERAGE, GENE, RESISTANCE,
    DATABASE. Returns the output path.
    """
    if db != "all":
        raise NotImplementedError(
            "This module screens against all four bundled databases "
            "(CARD, ResFinder, MEGARes, AMRFinderPlus) combined; "
            "per-database screening isn't implemented."
        )

    _ensure_blast_available()
    databases = _get_databases()

    for entry in databases:
        entry["db_path"] = _build_blast_db_if_needed(entry["fasta"], entry["db_path"])

    # best_hits[qseqid] = {"bitscore", "pident", "length", "qlen", "gene", "resistance", "database"}
    best_hits = {}

    for entry in databases:
        rows = _blastn(genes_fasta, entry["db_path"])
        for row in rows:
            parsed = entry["parse_fn"](row["sseqid"])
            if parsed is None:
                continue  # filtered out (wrong type/category for this database)
            gene_name, resistance = parsed

            qseqid = row["qseqid"]
            bitscore = float(row["bitscore"])
            current_best = best_hits.get(qseqid)
            if current_best is None or bitscore > current_best["bitscore"]:
                best_hits[qseqid] = {
                    "bitscore": bitscore,
                    "pident": float(row["pident"]),
                    "length": float(row["length"]),
                    "qlen": float(row["qlen"]),
                    "gene": gene_name,
                    "resistance": resistance,
                    "database": entry["name"],
                }

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["SEQUENCE", "%IDENTITY", "%COVERAGE", "GENE", "RESISTANCE", "DATABASE"])
        for qseqid, hit in best_hits.items():
            coverage = round(hit["length"] / hit["qlen"] * 100, 2) if hit["qlen"] else 0.0
            writer.writerow([qseqid, hit["pident"], coverage, hit["gene"], hit["resistance"], hit["database"]])

    return str(output_csv)


def parse_screening_results(blast_output: str) -> list:
    """
    Parse the TSV written by run_arg_screening() into a list of dicts with
    gene, identity, coverage, resistance_class, matched_database, and a
    simple similarity call (High/Moderate/Low) based on %identity.
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
                "matched_database": row.get("DATABASE", "Unknown"),
                "result": "Known ARG" if similarity == "High" else
                          "Candidate ARG" if similarity == "Moderate" else "Unknown",
            })
    return hits


if __name__ == "__main__":
    import sys
    out = run_arg_screening(sys.argv[1], sys.argv[2])
    print(parse_screening_results(out))
