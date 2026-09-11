"""
scripts/build_negative_set.py  (v2)

Builds the NEGATIVE (non-ARG) training set using real bacterial genes
rather than random or synthetic DNA — this matters scientifically because
a classifier trained against random sequences learns to separate
"biological gene" from "random string" (trivial, driven by codon usage
and ORF structure alone) rather than "resistance gene" from "ordinary
gene" (the actual task). Real negative genes force the model to learn
genuine ARG-specific signal.

Pipeline per genome:
    NCBI RefSeq genome download
        -> Prodigal gene prediction
        -> sequential screening: CARD -> AMRFinderPlus -> ResFinder
        -> keep only genes with NO hit in ANY of the three databases
    (repeated across 12-15 phylogenetically diverse genomes)
        -> pool all no-hit genes
        -> CD-HIT dedup (90% identity, consistent with the positive set)
        -> randomly subsample down to roughly match positive set size
           (balanced dataset -> avoids a majority-class bias in training)

Why 12-15 genomes across diverse phyla (not 4)?
A negative set built from only 1-4 genomes over-represents that
organism's specific codon usage bias and GC content, so the model may
learn "genes with E. coli-like codon usage = non-ARG" rather than a
genuinely ARG-specific signal. Using genomes spanning multiple phyla
(Proteobacteria, Firmicutes, Actinobacteria, Bacteroidetes) exposes the
model to the natural diversity of bacterial gene composition, which is
the same diversity the positive (ARG) set already spans since ARGs
originate from many different organisms.

Why screen against CARD + AMRFinderPlus + ResFinder sequentially (not
just one)?
No single ARG database has perfect recall. A gene with no CARD hit might
still be a known resistance gene only catalogued in ResFinder or
AMRFinderPlus. Screening against only one database risks contaminating
the "negative" class with true, undetected ARGs, which would directly
corrupt model training (false negatives labeled as negatives). Requiring
NO hit across all three is a conservative, defensible negative-labeling
criterion appropriate for a dissertation methodology.

Requires (installed locally, NOT available in a sandboxed environment):
    conda install -c conda-forge ncbi-datasets-cli
    conda install -c bioconda prodigal abricate cd-hit
    abricate --setupdb          # installs ABRicate's bundled db set
    abricate --check            # confirm 'card', 'ncbi' (AMRFinderPlus-derived), 'resfinder' are present

Note on ABRicate database naming: ABRicate ships CARD, ResFinder, and an
NCBI-derived database (labelled 'ncbi', built from the same Reference
Gene Catalog AMRFinderPlus uses) as built-in options. We use these three
ABRicate databases as the sequential screen. (There is no separate
RGI/AMRFinderPlus-native screening path in this script — everything goes
through ABRicate's bundled databases.)

Usage:
    python scripts/build_negative_set.py --outdir data/training/negative
"""
import argparse
import csv
import random
import subprocess
import zipfile
from pathlib import Path

from Bio import SeqIO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, timed_step, ExternalToolError, get_progress_bar
from src.dataset_versioning import record_dataset_version

logger = get_logger(__name__)

# 15 phylogenetically diverse, complete, well-annotated RefSeq genomes
# spanning 5 major bacterial phyla. Swap/extend freely — diversity matters
# more than these exact accessions.
DEFAULT_ACCESSIONS = [
    # Proteobacteria
    "GCF_000005845.2",   # Escherichia coli K-12 MG1655
    "GCF_000006945.2",   # Salmonella enterica Typhimurium LT2
    "GCF_000006765.1",   # Pseudomonas aeruginosa PAO1
    "GCF_000009605.1",   # Helicobacter pylori 26695
    "GCF_000007765.2",   # Vibrio cholerae N16961
    # Firmicutes
    "GCF_000009045.1",   # Bacillus subtilis 168
    "GCF_000013425.1",   # Staphylococcus aureus NCTC 8325
    "GCF_000007785.1",   # Listeria monocytogenes EGD-e
    "GCF_000271365.1",   # Enterococcus faecalis
    # Actinobacteria
    "GCF_000195955.2",   # Mycobacterium tuberculosis H37Rv
    "GCF_000196335.1",   # Streptomyces coelicolor A3(2)
    "GCF_000009985.1",   # Corynebacterium glutamicum
    # Bacteroidetes
    "GCF_000025985.1",   # Bacteroides fragilis
    # Spirochaetes / Cyanobacteria (extra diversity)
    "GCF_000008685.2",   # Treponema pallidum
    "GCF_000009725.1",   # Synechocystis sp. PCC 6803
]

ABRICATE_DBS = ["card", "ncbi", "resfinder"]   # sequential 3-database screen


@timed_step("Download RefSeq genomes")
def download_genomes(accessions: list, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for acc in get_progress_bar(accessions, desc="Downloading genomes"):
        target_zip = dest_dir / f"{acc}.zip"
        if (dest_dir / acc).exists():
            logger.info(f"{acc} already downloaded, skipping")
            continue
        try:
            subprocess.run([
                "datasets", "download", "genome", "accession", acc,
                "--include", "genome", "--filename", str(target_zip),
            ], check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise ExternalToolError(
                "NCBI 'datasets' CLI not found. Install with: "
                "conda install -c conda-forge ncbi-datasets-cli"
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to download {acc}: {e.stderr}. Skipping this genome.")
            continue

        # Extract with Python's built-in zipfile — no dependency on an
        # external `unzip` binary, which isn't available on a plain Windows
        # install (this was the original bug: the previous version of this
        # function assumed `unzip` was on PATH).
        try:
            with zipfile.ZipFile(target_zip, "r") as zip_ref:
                zip_ref.extractall(dest_dir / acc)
        except zipfile.BadZipFile:
            # A corrupt/partial download (e.g. connection dropped mid-
            # transfer) leaves a .zip that fails to open. Don't leave a
            # half-extracted directory behind — remove both the bad zip
            # and any partial extraction dir so a re-run retries cleanly
            # instead of silently treating it as "already downloaded".
            logger.warning(f"{acc}: downloaded zip is corrupt/incomplete, skipping this genome. "
                            "Re-run the script to retry — the corrupt file has been removed.")
            target_zip.unlink(missing_ok=True)
            extracted_dir = dest_dir / acc
            if extracted_dir.exists():
                import shutil
                shutil.rmtree(extracted_dir)
            continue
        finally:
            # The extracted .fna is what we actually need; the zip itself
            # is just an intermediate download artifact.
            target_zip.unlink(missing_ok=True)


def find_genome_fasta(genome_dir: Path) -> Path:
    matches = list(genome_dir.rglob("*.fna"))
    if not matches:
        raise FileNotFoundError(f"No .fna file found under {genome_dir}")
    return matches[0]


@timed_step("Run Prodigal")
def run_prodigal(genome_fasta: Path, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            "prodigal", "-i", str(genome_fasta),
            "-d", str(out_prefix.with_suffix(".fna")),
            "-a", str(out_prefix.with_suffix(".faa")),
            "-o", str(out_prefix.with_suffix(".gff")), "-f", "gff",
        ], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise ExternalToolError("Prodigal not found. Install with: conda install -c bioconda prodigal")
    except subprocess.CalledProcessError as e:
        raise ExternalToolError(f"Prodigal failed on {genome_fasta}:\n{e.stderr}")


def run_abricate(genes_fasta: Path, db: str, out_tsv: Path):
    try:
        result = subprocess.run(["abricate", "--db", db, str(genes_fasta)],
                                 check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise ExternalToolError("ABRicate not found. Install with: conda install -c bioconda -c conda-forge abricate")
    except subprocess.CalledProcessError as e:
        raise ExternalToolError(f"ABRicate ({db}) failed:\n{e.stderr}")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text(result.stdout)


def get_hit_ids(abricate_tsv: Path) -> set:
    if not abricate_tsv.exists():
        return set()
    ids = set()
    with open(abricate_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.add(row["SEQUENCE"])
    return ids


@timed_step("Sequential 3-database screening")
def screen_against_all_databases(genes_nt_fasta: Path, work_dir: Path, genome_acc: str) -> set:
    """
    Runs ABRicate against CARD, then AMRFinderPlus-derived ('ncbi'), then
    ResFinder. Returns the UNION of all hit gene IDs — any gene that hits
    ANY of the three is excluded from the negative set.
    """
    all_hits = set()
    for db in ABRICATE_DBS:
        out_tsv = work_dir / "screening" / f"{genome_acc}_{db}.tsv"
        run_abricate(genes_nt_fasta, db, out_tsv)
        hits = get_hit_ids(out_tsv)
        logger.info(f"  {genome_acc} vs {db}: {len(hits)} genes hit")
        all_hits |= hits
    return all_hits


@timed_step("CD-HIT dedup of pooled negative genes")
def run_cdhit(input_fasta: Path, output_fasta: Path, identity: float = 0.90):
    try:
        subprocess.run([
            "cd-hit-est", "-i", str(input_fasta), "-o", str(output_fasta),
            "-c", str(identity), "-n", "8", "-M", "4000", "-T", "0",
        ], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise ExternalToolError("cd-hit-est not found. Install with: conda install -c bioconda cd-hit")
    except subprocess.CalledProcessError as e:
        raise ExternalToolError(f"CD-HIT failed:\n{e.stderr}")


def subsample_fasta(input_fasta: Path, output_fasta: Path, target_count: int, seed: int = 42):
    """Randomly subsample down to target_count sequences for class balance."""
    records = list(SeqIO.parse(str(input_fasta), "fasta"))
    if len(records) <= target_count:
        logger.info(f"Only {len(records)} negative sequences available "
                     f"(target {target_count}) — keeping all, dataset will be mildly imbalanced.")
        SeqIO.write(records, str(output_fasta), "fasta")
        return len(records)

    random.seed(seed)
    sampled = random.sample(records, target_count)
    SeqIO.write(sampled, str(output_fasta), "fasta")
    return target_count


def main(accessions: list, outdir: Path, work_dir: Path, identity: float, target_count: int):
    outdir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    pooled_nt = work_dir / "pooled_negative_raw.fna"
    pooled_nt.write_text("")

    download_genomes(accessions, work_dir / "genomes")

    total_pooled = 0
    for acc in get_progress_bar(accessions, desc="Screening genomes"):
        genome_dir = work_dir / "genomes" / acc
        if not genome_dir.exists():
            continue
        try:
            genome_fasta = find_genome_fasta(genome_dir)
        except FileNotFoundError:
            logger.warning(f"Skipping {acc}: genome FASTA not found (download may have failed)")
            continue

        gene_prefix = work_dir / "genes" / acc
        run_prodigal(genome_fasta, gene_prefix)

        genes_nt = gene_prefix.with_suffix(".fna")
        hit_ids = screen_against_all_databases(genes_nt, work_dir, acc)

        kept = 0
        with open(pooled_nt, "a") as out_handle:
            for record in SeqIO.parse(str(genes_nt), "fasta"):
                if record.id not in hit_ids:
                    record.id = f"{acc}__{record.id}"
                    record.description = f"source=RefSeq_negative|genome={acc}"
                    SeqIO.write(record, out_handle, "fasta")
                    kept += 1
        logger.info(f"{acc}: {kept} confirmed non-ARG genes pooled")
        total_pooled += kept

    logger.info(f"Total pooled (pre-dedup): {total_pooled} genes")

    deduped = outdir / "negative_dedup.fasta"
    run_cdhit(pooled_nt, deduped, identity=identity)

    final_fasta = outdir / "negative_final.fasta"
    final_count = subsample_fasta(deduped, final_fasta, target_count)

    record_dataset_version(
        source_db="NCBI_RefSeq_negative_genomes",
        num_downloaded=total_pooled,
        num_retained=final_count,
        db_version=f"RefSeq genomes as of {__import__('datetime').date.today().isoformat()} "
                    f"({len(accessions)} accessions, screened against {', '.join(ABRICATE_DBS)})",
        notes="num_downloaded = genes pooled after 3-database no-hit screening, pre-CD-HIT; "
              "num_retained = final count after CD-HIT dedup + subsampling for class balance.",
    )
    logger.info("Dataset version log updated -> reports/dataset_versions.csv")

    logger.info(f"Final negative dataset: {final_count} sequences -> {final_fasta}")
    logger.info("Run scripts/prepare_dataset.py next to merge with the positive set and split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the negative (non-ARG) training set from real bacterial genomes")
    parser.add_argument("--outdir", default="data/training/negative")
    parser.add_argument("--work-dir", default="data/raw_negative_build")
    parser.add_argument("--accessions", nargs="+", default=DEFAULT_ACCESSIONS)
    parser.add_argument("--identity", type=float, default=0.90, help="CD-HIT identity threshold (match positive set)")
    parser.add_argument("--target-count", type=int, default=3000,
                         help="Subsample to roughly this many sequences to balance against the positive set "
                              "(check data/training/positive/positive_labels.csv row count and set accordingly)")
    args = parser.parse_args()

    main(args.accessions, Path(args.outdir), Path(args.work_dir), args.identity, args.target_count)
