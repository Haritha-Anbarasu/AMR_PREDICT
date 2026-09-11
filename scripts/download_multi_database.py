"""
download_multi_database.py

Automatically downloads and labels ARG reference sequences from FOUR
databases — the combination commonly used together for a robust ARG
prediction pipeline:

    1. CARD               (card.mcmaster.ca)
    2. NCBI AMRFinderPlus  (ftp.ncbi.nlm.nih.gov)
    3. ResFinder           (bitbucket.org/genomicepidemiology)
    4. MEGARes             (megares.meglab.org)

Each database has its own function (download_card, download_amrfinderplus,
download_resfinder, download_megares) so you can run all four or pick a
subset. A final merge step combines everything into one nucleotide FASTA +
one labels CSV, ready for src/sequence_features.py, src/protein_features.py,
and src/cluster_split.py.

Requires internet access — run this on your own machine, not in a
sandboxed/offline environment. Requires: pandas, biopython, and `git`
on PATH (for ResFinder).

Usage:
    # Run all four databases
    python scripts/download_multi_database.py --outdir data/training

    # Run only specific databases
    python scripts/download_multi_database.py --outdir data/training --databases card resfinder

Output (in --outdir):
    card/card_nucleotide.fasta, card/card_labels.csv
    amrfinderplus/amrfinderplus_nucleotide.fasta, amrfinderplus/amrfinderplus_labels.csv
    resfinder/resfinder_nucleotide.fasta, resfinder/resfinder_labels.csv
    megares/megares_nucleotide.fasta, megares/megares_labels.csv
    merged_arg_dataset.fasta   <- combined, deduplicated-by-ID nucleotide sequences
    merged_arg_labels.csv      <- combined labels with a 'source_db' column
"""
import argparse
import re
import shutil
import ssl
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import pandas as pd
from Bio import SeqIO


def _download_with_retries(url: str, dest: Path, attempts: int = 4, base_delay: float = 2.0):
    """urlretrieve wrapper that retries on transient network/TLS errors.

    Large downloads over flaky Wi-Fi/VPN/proxies can hit errors like
    'SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC' or connection resets mid
    transfer. These are almost always transient — a fresh connection on
    retry usually succeeds. Any partially-written file from a failed
    attempt is removed before retrying so we never keep a corrupt file.
    """
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except (ssl.SSLError, ConnectionError, TimeoutError, OSError) as e:
            last_err = e
            if dest.exists():
                dest.unlink()
            if attempt < attempts:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"  Retry {attempt}/{attempts - 1} after error ({e}); "
                      f"waiting {delay:.0f}s ...")
                time.sleep(delay)
    raise last_err


# ----------------------------------------------------------------------
# 1. CARD
# ----------------------------------------------------------------------
CARD_DATA_URL = "https://card.mcmaster.ca/latest/data"
CARD_NT_FASTA_NAME = "nucleotide_fasta_protein_homolog_model.fasta"
CARD_ARO_INDEX_NAME = "aro_index.tsv"
ARO_PATTERN = re.compile(r"ARO:(\d+)")


def download_card(raw_dir: Path, outdir: Path):
    print("\n=== CARD ===")
    raw_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    tarball_path = raw_dir / "card-data.tar.bz2"
    print(f"Downloading {CARD_DATA_URL} ...")
    _download_with_retries(CARD_DATA_URL, tarball_path)
    with tarfile.open(tarball_path, "r:bz2") as tar:
        tar.extractall(raw_dir, filter="data")

    nt_fasta = raw_dir / CARD_NT_FASTA_NAME
    aro_index = raw_dir / CARD_ARO_INDEX_NAME
    if not nt_fasta.exists() or not aro_index.exists():
        print(f"  WARNING: expected files not found in {raw_dir}. "
              f"Run `ls {raw_dir}` to see what CARD actually shipped and "
              "update CARD_NT_FASTA_NAME / CARD_ARO_INDEX_NAME above.")
        return None

    aro_df = pd.read_csv(aro_index, sep="\t")
    aro_df.columns = [c.strip() for c in aro_df.columns]
    aro_df["aro_id"] = aro_df["ARO Accession"].str.extract(r"ARO:(\d+)")

    rows = []
    with open(nt_fasta) as f:
        for line in f:
            if line.startswith(">"):
                header = line[1:].strip()
                match = ARO_PATTERN.search(header)
                rows.append({
                    "gene_id": header.split(" ")[0],
                    "aro_id": match.group(1) if match else None,
                })
    headers_df = pd.DataFrame(rows)
    merged = headers_df.merge(aro_df, on="aro_id", how="left")

    labels = pd.DataFrame({
        "gene_id": merged["gene_id"],
        "drug_class": merged.get("Drug Class"),
        "gene_family": merged.get("AMR Gene Family"),
        "resistance_mechanism": merged.get("Resistance Mechanism"),
        "source_db": "CARD",
    })

    # Guarantee uniqueness: header.split(" ")[0] gives the real per-sequence
    # token from the FASTA header (e.g. "gb|AE006468.1|-|3194809-3195988|ARO:
    # 3003345"), which is unique in the vast majority of CARD entries. But
    # CARD's homolog-model FASTA occasionally repeats an identical header for
    # closely related variant entries, so we still enforce uniqueness
    # explicitly rather than assuming the header format guarantees it. Any
    # duplicate gets a numeric suffix; the original id is fully preserved as
    # the prefix, so traceability back to the source header isn't lost.
    dup_mask = labels["gene_id"].duplicated(keep=False)
    if dup_mask.any():
        suffix_counter = labels.groupby("gene_id").cumcount()
        labels.loc[dup_mask, "gene_id"] = (
            labels.loc[dup_mask, "gene_id"] + "_dup" + (suffix_counter[dup_mask] + 1).astype(str)
        )

    nt_out = outdir / "card_nucleotide.fasta"
    shutil.copy(nt_fasta, nt_out)
    labels_out = outdir / "card_labels.csv"
    labels.to_csv(labels_out, index=False)
    print(f"  {len(labels)} sequences -> {nt_out}, {labels_out}")

    n_unique = labels["gene_id"].nunique()
    print(f"  gene_id uniqueness: {n_unique} unique out of {len(labels)} rows")
    if n_unique < len(labels):
        print(f"  WARNING: gene_id still not fully unique after de-duplication — "
              "this shouldn't happen; check the header.split(...) logic above "
              f"against a sample header in {nt_fasta}.")
    return nt_out, labels_out


# ----------------------------------------------------------------------
# 2. NCBI AMRFinderPlus
# ----------------------------------------------------------------------
AMRFINDER_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest/"
AMRFINDER_FILES = {
    "nucleotide": "AMR_CDS.fa",
    "protein": "AMRProt.fa",
    "catalog": "ReferenceGeneCatalog.txt",
}


def download_amrfinderplus(raw_dir: Path, outdir: Path):
    print("\n=== NCBI AMRFinderPlus ===")
    raw_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    local_paths = {}
    for key, filename in AMRFINDER_FILES.items():
        url = AMRFINDER_BASE_URL + filename
        dest = raw_dir / filename
        print(f"Downloading {url} ...")
        try:
            _download_with_retries(url, dest)
            local_paths[key] = dest
        except Exception as e:
            print(f"  WARNING: could not download {filename}: {e}")

    if "nucleotide" not in local_paths or "catalog" not in local_paths:
        print("  WARNING: required files missing, skipping AMRFinderPlus.")
        return None

    catalog = pd.read_csv(local_paths["catalog"], sep="\t", low_memory=False)
    catalog.columns = [c.strip() for c in catalog.columns]

    # AMR_CDS.fa deflines are pipe-delimited compound records, NOT a bare
    # accession:
    #   field 1: protein accession
    #   field 2: DNA (nucleotide) accession
    #   field 3-4: fusion gene part / total parts
    #   field 5-6: node_id / parent node_id
    #   field 7: resistance mechanism type
    #   field 8: protein name
    # Biopython's record.id captures the *entire* pipe-delimited string
    # (there's no whitespace before the first space, which comes after
    # field 8). Matching that whole compound string against a single
    # accession column in the catalog can never succeed — that was the
    # actual bug causing every label to come back blank. The fix: split
    # the header ourselves and match the protein/DNA accession sub-fields
    # against the catalog's accession columns.
    protein_acc_cols = [c for c in ["refseq_protein_accession", "genbank_protein_accession"]
                        if c in catalog.columns]
    nuc_acc_cols = [c for c in ["refseq_nucleotide_accession", "genbank_nucleotide_accession"]
                    if c in catalog.columns]
    label_cols = [c for c in ["gene_family", "class", "subclass", "type", "subtype"]
                  if c in catalog.columns]

    def _build_lookup(acc_cols):
        frames = []
        for acc_col in acc_cols:
            sub = (
                catalog[[acc_col] + label_cols]
                .dropna(subset=[acc_col])
                .rename(columns={acc_col: "accession"})
            )
            frames.append(sub)
        if frames:
            return pd.concat(frames, ignore_index=True).drop_duplicates(subset="accession")
        return pd.DataFrame(columns=["accession"] + label_cols)

    protein_lookup = _build_lookup(protein_acc_cols)
    nuc_lookup = _build_lookup(nuc_acc_cols)

    def _strip_version(acc):
        # "AAA16360.1" -> "AAA16360". Accession version suffixes sometimes
        # differ between AMR_CDS.fa headers and the catalog (e.g. a header
        # was generated against a newer/older record version than what's
        # currently in ReferenceGeneCatalog.txt), which silently breaks an
        # exact-string match even though it's clearly the same accession.
        if not isinstance(acc, str):
            return acc
        return acc.split(".")[0]

    rows = []
    for record in SeqIO.parse(str(local_paths["nucleotide"]), "fasta"):
        parts = record.id.split("|")
        rows.append({
            "gene_id": record.id,
            "protein_acc": parts[0] if len(parts) > 0 else None,
            "dna_acc": parts[1] if len(parts) > 1 else None,
        })
    seq_df = pd.DataFrame(rows)

    # Pass 1: exact protein accession match.
    merged = seq_df.merge(protein_lookup, left_on="protein_acc", right_on="accession", how="left")
    unmatched = merged[label_cols].isna().all(axis=1) if label_cols else pd.Series(True, index=merged.index)

    # Pass 2: exact DNA accession match, for whatever pass 1 missed.
    if unmatched.any() and not nuc_lookup.empty:
        fallback = seq_df.loc[unmatched, ["dna_acc"]].merge(
            nuc_lookup, left_on="dna_acc", right_on="accession", how="left"
        )
        for col in label_cols:
            merged.loc[unmatched, col] = fallback[col].values
        unmatched = merged[label_cols].isna().all(axis=1) if label_cols else pd.Series(True, index=merged.index)

    # Pass 3: version-stripped protein accession match, for anything still
    # unmatched — catches cases where the FASTA header and catalog disagree
    # only on the trailing ".N" version suffix.
    if unmatched.any() and not protein_lookup.empty:
        protein_lookup_noversion = protein_lookup.copy()
        protein_lookup_noversion["accession"] = protein_lookup_noversion["accession"].map(_strip_version)
        protein_lookup_noversion = protein_lookup_noversion.drop_duplicates(subset="accession")
        fallback_acc = seq_df.loc[unmatched, "protein_acc"].map(_strip_version)
        fallback = fallback_acc.to_frame(name="protein_acc_noversion").merge(
            protein_lookup_noversion, left_on="protein_acc_noversion", right_on="accession", how="left"
        )
        for col in label_cols:
            merged.loc[unmatched, col] = fallback[col].values

    labels = pd.DataFrame({
        "gene_id": merged["gene_id"],
        "drug_class": merged.get("class"),
        "gene_family": merged.get("gene_family"),
        "resistance_mechanism": merged.get("subtype"),
        "element_type": merged.get("type"),
        "source_db": "AMRFinderPlus",
    })

    # AMRFinderPlus's ReferenceGeneCatalog.txt covers more than antibiotic
    # resistance: alongside "type" == "AMR" it also carries "VIRULENCE"
    # (T3SS effectors, toxins, siderophore receptors, fimbriae...),
    # "STRESS"/"METAL"/"BIOCIDE"/"HEAT"/"ACID" (metal/biocide/stress
    # tolerance genes), etc. Those genes never have a "class" (antibiotic
    # drug class) value — that's expected, not a matching failure — and
    # more importantly they are NOT antibiotic resistance genes at all.
    # Previously this function copied AMR_CDS.fa and wrote labels.csv for
    # *every* catalog entry unfiltered, so build_positive_dataset.py went
    # on to label virulence/stress genes "ARG" with drug_class="unknown".
    # That's dataset contamination, not a missing-label display issue —
    # filter to type == "AMR" here so only genuine resistance genes ever
    # reach the positive ARG set.
    if "element_type" in labels.columns and labels["element_type"].notna().any():
        type_counts = labels["element_type"].value_counts(dropna=False)
        print(f"  AMRFinderPlus element types found: {type_counts.to_dict()}")
        keep_mask = labels["element_type"] == "AMR"
        n_dropped = (~keep_mask).sum()
        if n_dropped:
            print(f"  Dropping {n_dropped} non-AMR entries (VIRULENCE/STRESS/METAL/BIOCIDE/etc.) "
                  "— these are not antibiotic resistance genes and don't belong in the ARG positive set.")
        labels = labels.loc[keep_mask].reset_index(drop=True)
        keep_ids = set(labels["gene_id"])
    else:
        print("  WARNING: could not find AMRFinderPlus 'type' column — cannot filter to AMR-only "
              "genes. All catalog entries (including virulence/stress genes, if present) will be "
              "kept. Check that ReferenceGeneCatalog.txt still has a 'type' column.")
        keep_ids = set(labels["gene_id"])

    labels = labels.drop(columns=["element_type"])

    nt_out = outdir / "amrfinderplus_nucleotide.fasta"
    n_written = 0
    with open(nt_out, "w") as out_handle:
        for record in SeqIO.parse(str(local_paths["nucleotide"]), "fasta"):
            if record.id in keep_ids:
                SeqIO.write(record, out_handle, "fasta")
                n_written += 1
    labels_out = outdir / "amrfinderplus_labels.csv"
    labels.to_csv(labels_out, index=False)
    print(f"  {len(labels)} sequences -> {nt_out} ({n_written} seqs written), {labels_out}")

    matched = labels["drug_class"].notna().sum() if "drug_class" in labels else 0
    match_pct = 100 * matched / len(labels) if len(labels) else 0
    print(f"  drug_class matched for {matched}/{len(labels)} sequences ({match_pct:.1f}%)")

    if match_pct < 50:
        print("  WARNING: match rate is low — AMRFinderPlus may have renamed catalog "
              f"columns again. Open {local_paths['catalog']} and check that "
              "genbank_protein_accession / refseq_protein_accession still exist "
              "and adjust protein_acc_cols/nuc_acc_cols/label_cols above.")
    return nt_out, labels_out


# ----------------------------------------------------------------------
# 3. ResFinder
# ----------------------------------------------------------------------
RESFINDER_REPO_URL = "https://bitbucket.org/genomicepidemiology/resfinder_db.git"

# ResFinder headers follow "{geneName}_{alleleNumber}_{accession}", e.g.
# "aac(6')-Ib_1_M23634" or "blaTEM-1_1_AY458016". The gene name portion
# (everything before the trailing "_<allele number>_<accession>") is the
# natural "gene family" label for this database — ResFinder doesn't ship a
# separate gene-family field the way CARD does, so we derive it from the
# header itself instead of leaving it blank.
RESFINDER_GENE_ID_PATTERN = re.compile(r"^(?P<gene_family>.+)_(?P<allele>\d+)_(?P<accession>[A-Za-z0-9_.\-]+)$")


def download_resfinder(raw_dir: Path, outdir: Path):
    print("\n=== ResFinder ===")
    raw_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    repo_dir = raw_dir / "resfinder_db"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print(f"Cloning {RESFINDER_REPO_URL} ...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", RESFINDER_REPO_URL, str(repo_dir)],
                        check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("  WARNING: git is not installed. Install git or download the repo manually.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: git clone failed: {e.stderr}")
        return None

    # ResFinder ships one .fsa file per antibiotic class (aminoglycoside.fsa,
    # beta-lactam.fsa, etc.) PLUS a convenience "all.fsa" that just
    # concatenates every gene from every class into one file (used
    # internally for building a single combined KMA/BLAST index). Including
    # it here would duplicate every sequence under the meaningless
    # drug_class "all" on top of its real, correct class label — that's the
    # bug that caused drug_class == "all" rows to appear.
    fsa_files = [f for f in repo_dir.glob("*.fsa") if f.stem.lower() != "all"]
    excluded = [f for f in repo_dir.glob("*.fsa") if f.stem.lower() == "all"]
    if excluded:
        print(f"  Skipping aggregate file(s) {[f.name for f in excluded]} "
              "(redundant with the per-class files, would mislabel drug_class as 'all').")
    if not fsa_files:
        print(f"  WARNING: no .fsa files found in {repo_dir}.")
        return None

    # ResFinder splits sequences into one file per antibiotic class —
    # the filename itself IS the drug class label.
    nt_out = outdir / "resfinder_nucleotide.fasta"
    rows = []
    with open(nt_out, "w") as out_handle:
        for fsa in fsa_files:
            drug_class = fsa.stem  # e.g. "beta-lactam", "aminoglycoside"
            for record in SeqIO.parse(str(fsa), "fasta"):
                SeqIO.write(record, out_handle, "fasta")
                match = RESFINDER_GENE_ID_PATTERN.match(record.id)
                rows.append({
                    "gene_id": record.id,
                    "drug_class": drug_class,
                    "gene_family": match.group("gene_family") if match else None,
                    "resistance_mechanism": None,
                    "source_db": "ResFinder",
                })

    labels = pd.DataFrame(rows)

    # ResFinder also ships a repo-root "phenotypes.txt" documenting, per
    # gene, the mechanism of resistance (this is a real field the ResFinder
    # authors curate — see Bortolaia et al. 2020 — it just isn't in the FASTA
    # headers). Its accession column values match gene_id exactly (both are
    # the full "geneName_alleleNumber_accession" token), so we can join
    # directly. Column names have shifted slightly across ResFinder releases,
    # so we match by substring instead of a hardcoded exact name.
    phenotypes_path = repo_dir / "phenotypes.txt"
    if phenotypes_path.exists():
        try:
            pheno_df = pd.read_csv(phenotypes_path, sep="\t")
            pheno_df.columns = [c.strip() for c in pheno_df.columns]
            acc_col = next((c for c in pheno_df.columns if "accession" in c.lower()), None)
            mech_col = next((c for c in pheno_df.columns if "mechanism" in c.lower()), None)
            if acc_col and mech_col:
                mech_lookup = (
                    pheno_df[[acc_col, mech_col]]
                    .dropna(subset=[acc_col])
                    .drop_duplicates(subset=acc_col)
                    .rename(columns={acc_col: "gene_id", mech_col: "resistance_mechanism"})
                )
                labels = labels.drop(columns=["resistance_mechanism"]).merge(
                    mech_lookup, on="gene_id", how="left"
                )
                # merge() appends resistance_mechanism at the end; restore the
                # documented column order (gene_id, drug_class, gene_family,
                # resistance_mechanism, source_db).
                labels = labels[["gene_id", "drug_class", "gene_family",
                                  "resistance_mechanism", "source_db"]]
            else:
                print(f"  WARNING: could not find accession/mechanism columns in "
                      f"{phenotypes_path} (columns seen: {list(pheno_df.columns)}). "
                      "resistance_mechanism will stay blank.")
        except Exception as e:
            print(f"  WARNING: failed to parse {phenotypes_path} ({e}); "
                  "resistance_mechanism will stay blank.")
    else:
        print(f"  WARNING: {phenotypes_path} not found in cloned repo; "
              "resistance_mechanism will stay blank.")

    labels_out = outdir / "resfinder_labels.csv"
    labels.to_csv(labels_out, index=False)
    print(f"  {len(labels)} sequences -> {nt_out}, {labels_out}")

    mech_matched = labels["resistance_mechanism"].notna().sum() if "resistance_mechanism" in labels else 0
    mech_pct = 100 * mech_matched / len(labels) if len(labels) else 0
    print(f"  resistance_mechanism populated for {mech_matched}/{len(labels)} sequences ({mech_pct:.1f}%)")

    matched = labels["gene_family"].notna().sum()
    match_pct = 100 * matched / len(labels) if len(labels) else 0
    print(f"  gene_family parsed for {matched}/{len(labels)} sequences ({match_pct:.1f}%)")
    if match_pct < 90:
        print("  WARNING: gene_family parse rate is low — some ResFinder headers may not "
              "follow the usual 'geneName_alleleNumber_accession' pattern. Inspect a few "
              f"unmatched gene_id values in {labels_out} and adjust RESFINDER_GENE_ID_PATTERN "
              "above if needed.")

    n_unique = labels["gene_id"].nunique()
    if n_unique < len(labels):
        dup_classes = labels.loc[labels["gene_id"].duplicated(keep=False), "drug_class"].unique()
        print(f"  WARNING: {len(labels) - n_unique} duplicate gene_id row(s) found across "
              f"per-class files (classes involved: {list(dup_classes)}). If 'all' appears "
              "here, the aggregate-file exclusion above may need updating for a new ResFinder "
              "filename convention.")
    return nt_out, labels_out


# ----------------------------------------------------------------------
# 4. MEGARes
# ----------------------------------------------------------------------
MEGARES_VERSION = "3.00"
MEGARES_BASE_URL = f"https://www.meglab.org/downloads/megares_v{MEGARES_VERSION}/"
# NOTE: as of 2026 the old megares.meglab.org host no longer serves files —
# it 302-redirects to an HTML landing page on meglab.org, which is why this
# used to download successfully (HTTP 200) but fail to parse as FASTA. The
# filename also changed slightly: "megares_database" (not "_full_database").
MEGARES_FASTA_NAME = f"megares_database_v{MEGARES_VERSION}.fasta"


def _strip_fasta_comment_lines(path: Path):
    """Drop any lines before the first '>' record so Biopython's strict
    'fasta' parser (used here and later in merge_datasets) can read the
    file. MEGARes ships a comment header block before the first sequence;
    the offending lines aren't guaranteed to start with ';', so instead of
    filtering by prefix we just find the first '>' and keep everything
    from there onward."""
    with open(path) as f:
        lines = f.readlines()

    first_seq_idx = next((i for i, line in enumerate(lines) if line.lstrip().startswith(">")), None)
    if first_seq_idx is None:
        print(f"  WARNING: no '>' header found in {path}; leaving file untouched.")
        return

    if first_seq_idx != 0:
        with open(path, "w") as f:
            f.writelines(lines[first_seq_idx:])


def download_megares(raw_dir: Path, outdir: Path):
    print("\n=== MEGARes ===")
    raw_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    url = MEGARES_BASE_URL + MEGARES_FASTA_NAME
    dest = raw_dir / MEGARES_FASTA_NAME
    print(f"Downloading {url} ...")
    try:
        _download_with_retries(url, dest)
    except Exception as e:
        print(f"  WARNING: could not download MEGARes v{MEGARES_VERSION} "
              f"({e}). Check https://www.meglab.org/megares/download/ "
              "for the current version/filename and update MEGARES_VERSION "
              "/ MEGARES_FASTA_NAME above.")
        return None

    # Sanity-check: a stale/redirected URL will happily return HTTP 200 with
    # an HTML landing page instead of the real file, which is what caused
    # the confusing "FASTA file contains comments" parser error before.
    with open(dest, "rb") as f:
        head = f.read(512).lstrip()
    if not head.startswith(b">"):
        print(f"  WARNING: downloaded file does not look like FASTA (starts with "
              f"{head[:80]!r}). The site may have changed its download URL again — "
              "check https://www.meglab.org/megares/download/ and update "
              "MEGARES_BASE_URL / MEGARES_FASTA_NAME above.")
        return None

    _strip_fasta_comment_lines(dest)

    # MEGARes headers are self-describing:
    #   >MEG_1|Drugs|Aminoglycosides|Aminoglycoside_resistance_gene|AAC6-Ie-APH2-Ia
    # i.e. id|compound_type|drug_class|mechanism|gene_group
    rows = []
    nt_out = outdir / "megares_nucleotide.fasta"
    shutil.copy(dest, nt_out)

    for record in SeqIO.parse(str(dest), "fasta"):
        parts = record.description.split("|")
        rows.append({
            "gene_id": record.id,
            "drug_class": parts[2] if len(parts) > 2 else None,
            "gene_family": parts[4] if len(parts) > 4 else None,
            "resistance_mechanism": parts[3] if len(parts) > 3 else None,
            "source_db": "MEGARes",
        })

    labels = pd.DataFrame(rows)
    labels_out = outdir / "megares_labels.csv"
    labels.to_csv(labels_out, index=False)
    print(f"  {len(labels)} sequences -> {nt_out}, {labels_out}")
    return nt_out, labels_out


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------
def merge_datasets(results: dict, outdir: Path):
    print("\n=== Merging all downloaded databases ===")
    all_labels = []
    merged_fasta_path = outdir / "merged_arg_dataset.fasta"

    with open(merged_fasta_path, "w") as merged_handle:
        for db_name, result in results.items():
            if result is None:
                continue
            nt_path, labels_path = result
            labels_df = pd.read_csv(labels_path)

            # Prefix gene_id with the source DB to guarantee global uniqueness
            id_map = {}
            for record in SeqIO.parse(str(nt_path), "fasta"):
                new_id = f"{db_name}__{record.id}"
                id_map[record.id] = new_id
                record.id = new_id
                record.description = ""
                SeqIO.write(record, merged_handle, "fasta")

            labels_df["gene_id"] = labels_df["gene_id"].map(id_map).fillna(labels_df["gene_id"])
            all_labels.append(labels_df)

    if not all_labels:
        print("  Nothing to merge — no databases downloaded successfully.")
        return

    merged_labels = pd.concat(all_labels, ignore_index=True)
    merged_labels_path = outdir / "merged_arg_labels.csv"
    merged_labels.to_csv(merged_labels_path, index=False)

    print(f"  Total sequences merged: {len(merged_labels)}")
    print(f"  {merged_fasta_path}")
    print(f"  {merged_labels_path}")
    print("\nSequences per source database:")
    print(merged_labels["source_db"].value_counts().to_string())


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
DOWNLOADERS = {
    "card": download_card,
    "amrfinderplus": download_amrfinderplus,
    "resfinder": download_resfinder,
    "megares": download_megares,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and label ARG sequences from multiple databases")
    parser.add_argument("--outdir", default="data/training", help="Base output directory")
    parser.add_argument("--raw-dir", default="data/raw_downloads", help="Where raw downloads are extracted")
    parser.add_argument("--databases", nargs="+", choices=list(DOWNLOADERS.keys()),
                         default=list(DOWNLOADERS.keys()),
                         help="Which databases to download (default: all four)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    raw_dir = Path(args.raw_dir)

    results = {}
    for db_name in args.databases:
        db_raw_dir = raw_dir / db_name
        db_outdir = outdir / db_name
        try:
            results[db_name] = DOWNLOADERS[db_name](db_raw_dir, db_outdir)
        except Exception as e:
            print(f"\nERROR downloading {db_name}: {e}")
            results[db_name] = None

    merge_datasets(results, outdir)
