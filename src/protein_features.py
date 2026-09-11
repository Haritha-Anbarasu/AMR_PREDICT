"""
Module 5 (cont.): Protein-Based Feature Extraction

Computes protein-level physicochemical features from a translated gene
sequence: length, amino acid composition, molecular weight,
isoelectric point, hydrophobicity (GRAVY), aromaticity, instability
index, and aliphatic index.

Scientific rationale for each added feature (for viva):
- Aromaticity (fraction of Phe+Trp+Tyr residues): membrane-associated
  and efflux-pump proteins — a major ARG mechanism class (e.g.
  tetracycline/multidrug efflux pumps) — are typically enriched in
  aromatic residues relative to cytoplasmic enzymes, since aromatic
  side chains anchor transmembrane helices in the lipid bilayer.
- Instability index (Guruprasad et al. 1990 dipeptide-based measure):
  estimates whether a protein is likely stable in a test-tube/in-vivo
  context; horizontally-acquired resistance proteins expressed in a
  new host background can show different stability profiles than the
  host's native proteome, making this a plausible discriminating
  feature.
- Aliphatic index (Ikai 1980): the relative volume occupied by
  aliphatic side chains (Ala, Val, Ile, Leu) — a standard measure of
  thermostability, and known to vary systematically between enzyme
  functional classes (e.g. hydrolases vs. transferases), which
  β-lactamases/aminoglycoside-modifying enzymes and typical
  housekeeping genes may occupy differently.

All three are cheap to compute (already-parsed amino acid composition
plus two Biopython built-ins) and add biologically motivated signal
beyond composition alone.
"""
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Ikai (1980) aliphatic index coefficients
_ALIPHATIC_COEF_VAL = 2.9
_ALIPHATIC_COEF_ILE_LEU = 3.9


def _aliphatic_index(aa_percent: dict) -> float:
    """
    AI = X(Ala) + a*X(Val) + b*(X(Ile) + X(Leu))
    where X(*) are mole PERCENTAGES (0-100) and a=2.9, b=3.9.
    """
    ala = aa_percent.get("A", 0)
    val = aa_percent.get("V", 0)
    ile = aa_percent.get("I", 0)
    leu = aa_percent.get("L", 0)
    return ala + _ALIPHATIC_COEF_VAL * val + _ALIPHATIC_COEF_ILE_LEU * (ile + leu)


def extract_protein_features(protein_seq: str) -> dict:
    """
    Compute protein-level features for a single protein sequence.
    Sequences with non-standard residues (X, *, etc.) are cleaned first.
    """
    seq = "".join(ch for ch in protein_seq.upper() if ch in AMINO_ACIDS)
    if len(seq) == 0:
        empty = {
            "protein_length": 0, "molecular_weight": 0, "isoelectric_point": 0,
            "gravy_hydrophobicity": 0, "aromaticity": 0, "instability_index": 0,
            "aliphatic_index": 0,
        }
        empty.update({f"aa_{a}_pct": 0 for a in AMINO_ACIDS})
        return empty

    analysis = ProteinAnalysis(seq)
    # NOTE: Biopython's ProteinAnalysis API differs across versions —
    # older releases exposed get_amino_acids_percent() returning
    # fractions (0-1); Biopython >=1.81 exposes it as the
    # amino_acids_percent property, already scaled 0-100. We detect
    # which is available and normalize to a consistent 0-100 scale
    # either way, so this module works across Biopython versions
    # without silently producing wrong numbers on one of them.
    if hasattr(analysis, "amino_acids_percent"):
        raw = analysis.amino_acids_percent
        aa_percent_pct = {aa: raw.get(aa, 0) for aa in AMINO_ACIDS}   # already 0-100
    else:
        raw = analysis.get_amino_acids_percent()
        aa_percent_pct = {aa: raw.get(aa, 0) * 100 for aa in AMINO_ACIDS}   # 0-1 -> 0-100

    features = {
        "protein_length": len(seq),
        "molecular_weight": round(analysis.molecular_weight(), 2),
        "isoelectric_point": round(analysis.isoelectric_point(), 3),
        "gravy_hydrophobicity": round(analysis.gravy(), 4),
        "aromaticity": round(analysis.aromaticity(), 4),
        "instability_index": round(analysis.instability_index(), 3),
        "aliphatic_index": round(_aliphatic_index(aa_percent_pct), 3),
    }
    features.update({f"aa_{aa}_pct": round(aa_percent_pct[aa], 3) for aa in AMINO_ACIDS})
    return features


def extract_features_from_fasta(fasta_path: str) -> pd.DataFrame:
    """Build a protein feature matrix (one row per gene) from a protein multi-FASTA."""
    rows = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        feats = extract_protein_features(str(record.seq))
        feats["gene_id"] = record.id
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("gene_id")
    return df


if __name__ == "__main__":
    import sys
    df = extract_features_from_fasta(sys.argv[1])
    print(df.head())
