"""
src/label_harmonization.py

Maps the messy, database-specific drug_class strings in your positive
dataset (a mix of CARD, AMRFinderPlus, ResFinder, and MEGARes
vocabularies) onto a small set of canonical antibiotic drug-family
labels, and filters out any remaining non-antibiotic categories
(metal/biocide resistance) that slipped through upstream filters.

Why this matters scientifically: without harmonization, the exact same
biological drug class appears as multiple distinct labels purely due to
formatting differences between databases (e.g. 'MACROLIDE', 'macrolide',
'macrolide antibiotic' are three different strings for one class). A
multiclass model trained on unharmonized labels is being asked to
distinguish between classes that don't actually differ biologically —
this directly degrades accuracy and is not a modeling problem, it's a
data problem. Canonicalizing to drug FAMILY level (e.g. mapping
'carbapenem', 'cephalosporin', and 'penicillin' all to 'Beta-lactams')
also resolves the composite multi-drug-class strings common in MEGARes
and CARD annotations, which otherwise each become their own
near-unique, near-untrainable class.

Design choice — primary class vs. multi-label: many resistance genes
genuinely confer resistance to multiple drug families (efflux pumps
especially). This module extracts ALL matching canonical families per
entry (drug_class_all, semicolon-joined) AND a single primary class
(drug_class, the first canonical family matched) for straightforward
single-label multiclass training. Switching to true multi-label
classification later is possible using the drug_class_all column
without re-running any upstream pipeline stage — worth flagging as a
natural extension for a publication-track version of this project.
"""
import re

# Non-antibiotic categories that should never appear as a positive ARG
# drug_class — broader and more specific than the earlier keyword filter
# (which only matched "metal", missing specific element names).
OUT_OF_SCOPE_KEYWORDS = [
    "metal", "biocide", "copper", "mercury", "arsenic", "nickel", "silver",
    "chromium", "zinc", "cadmium", "aluminum", "aluminium", "iron",
    "cobalt", "tungsten", "sodium", "lead_resistance", "tellurium",
    "gold_resistance", "multi-compound", "multi_compound",
    "disinfect", "antiseptic", "phenolic_compound", "peroxide_resistance",
    "acid_resistance", "polyamine_resistance", "naphthoquinone",
    "quaternary_ammonium",
]

# Canonical drug family -> list of substrings (checked case-insensitively)
# that identify membership in that family. Order matters only in that
# the FIRST canonical family matched for a given entry becomes its
# "primary" label — families are checked in a fixed order below via
# CANONICAL_FAMILIES (a list, not a dict) to keep that behavior stable
# and reproducible across runs.
CANONICAL_FAMILIES = [
    ("Aminoglycosides", ["aminoglycoside"]),
    ("Beta-lactams", ["beta-lactam", "betalactam", "penicillin", "cephalosporin",
                       "carbapenem", "monobactam", "cephamycin"]),
    ("Macrolides", ["macrolide"]),
    ("Lincosamides", ["lincosamide"]),
    ("Streptogramins", ["streptogramin"]),
    ("Tetracyclines", ["tetracycline", "glycylcycline"]),
    ("Fluoroquinolones", ["fluoroquinolone", "quinolone"]),
    ("Sulfonamides", ["sulfonamide", "sulfone"]),
    ("Trimethoprim", ["trimethoprim", "diaminopyrimidine"]),
    ("Phenicols", ["phenicol", "chloramphenicol"]),
    ("Glycopeptides", ["glycopeptide"]),
    ("Oxazolidinones", ["oxazolidinone"]),
    ("Rifamycins", ["rifamycin", "rifampicin", "rifampin"]),
    ("Polymyxins", ["colistin", "polymyxin"]),
    ("Fosfomycin", ["fosfomycin", "phosphonic"]),
    ("Bacitracin", ["bacitracin"]),
    ("Mupirocin", ["mupirocin"]),
    ("Nitroimidazoles", ["nitroimidazole", "metronidazole"]),
    ("Pleuromutilins", ["pleuromutilin"]),
    ("Elfamycins", ["elfamycin"]),
    ("Nucleosides", ["nucleoside", "streptothricin", "pactamycin"]),
    ("Aminocoumarins", ["aminocoumarin"]),
    ("Fusidanes", ["fusidane", "fusidic"]),
    ("Lipopeptides", ["lipopeptide", "daptomycin"]),
    ("Bleomycins", ["bleomycin", "tetracenomycin"]),
    ("Moenomycins", ["moenomycin"]),
    ("Peptide_antibiotics", ["peptide antibiotic"]),
    ("Antitubercular", ["isoniazid"]),
    ("Spiropyrimidinetriones", ["spiropyrimidinetrione"]),
]


def is_out_of_scope(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in OUT_OF_SCOPE_KEYWORDS)


def harmonize_drug_class(raw: str) -> dict:
    """
    Given a raw drug_class string (possibly composite, e.g.
    'carbapenem;cephalosporin;monobactam' or inconsistently cased, e.g.
    'MACROLIDE'), returns:
        {
            "primary_class": str or None   -- first canonical family matched
            "all_classes": str or None     -- semicolon-joined ALL canonical families matched
            "out_of_scope": bool           -- True if this entry should be dropped entirely
        }
    Entries matching no canonical family return primary_class=None,
    which callers should treat as "unmapped — inspect manually" rather
    than silently dropping, so nothing disappears without a trace.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {"primary_class": None, "all_classes": None, "out_of_scope": False}

    if is_out_of_scope(raw):
        return {"primary_class": None, "all_classes": None, "out_of_scope": True}

    raw_lower = raw.lower()
    matched = []
    for canonical_name, keywords in CANONICAL_FAMILIES:
        if any(kw in raw_lower for kw in keywords):
            matched.append(canonical_name)

    if not matched:
        return {"primary_class": None, "all_classes": None, "out_of_scope": False}

    return {
        "primary_class": matched[0],
        "all_classes": ";".join(matched),
        "out_of_scope": False,
    }
