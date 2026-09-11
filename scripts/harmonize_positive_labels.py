"""
scripts/harmonize_positive_labels.py

Applies canonical drug-class mapping (src/label_harmonization.py) to
your EXISTING data/training/positive/positive_labels.csv in place —
no re-downloading or re-running the earlier pipeline stages needed.

What it does:
    1. Backs up the current file to positive_labels_pre_harmonization.csv
    2. Drops any remaining out-of-scope entries (metal/biocide resistance
       that slipped past the earlier filter — see src/label_harmonization.py
       docstring for why the earlier filter missed these)
    3. Maps every remaining drug_class value to a canonical drug family
       (drug_class column is overwritten with the canonical PRIMARY
       family; the original raw value is preserved in drug_class_raw;
       ALL matched families are preserved in drug_class_all for future
       multi-label work)
    4. Reports entries that matched NO canonical family (kept, unmapped —
       these are printed so you can inspect and, if appropriate, extend
       CANONICAL_FAMILIES in src/label_harmonization.py)
    5. Prints a before/after class-count summary

After running this, re-run (in order):
    python scripts/prepare_dataset.py
    python scripts/extract_features.py
    python scripts/train_models.py

Usage:
    python scripts/harmonize_positive_labels.py
"""
import argparse
import shutil
from pathlib import Path
from collections import Counter

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, timed_step
from src.label_harmonization import harmonize_drug_class

logger = get_logger(__name__)


@timed_step("Harmonize positive dataset drug class labels")
def main(labels_csv: Path):
    if not labels_csv.exists():
        raise FileNotFoundError(f"{labels_csv} not found. Run build_positive_dataset.py first.")

    backup_path = labels_csv.parent / "positive_labels_pre_harmonization.csv"
    if not backup_path.exists():
        shutil.copy(labels_csv, backup_path)
        logger.info(f"Backed up original labels -> {backup_path}")
    else:
        logger.info(f"Backup already exists at {backup_path} (not overwritten) — "
                     "using it as the source so re-running this script is idempotent.")

    df = pd.read_csv(backup_path)
    before_counts = df["drug_class"].value_counts()
    before_n_classes = df["drug_class"].nunique()
    before_n_rows = len(df)

    results = df["drug_class"].apply(harmonize_drug_class)

    df["drug_class_raw"] = df["drug_class"]
    df["drug_class"] = results.apply(lambda r: r["primary_class"])
    df["drug_class_all"] = results.apply(lambda r: r["all_classes"])
    out_of_scope_mask = results.apply(lambda r: r["out_of_scope"])

    n_out_of_scope = int(out_of_scope_mask.sum())
    df = df[~out_of_scope_mask].copy()
    logger.info(f"Dropped {n_out_of_scope} remaining out-of-scope (metal/biocide) entries")

    unmapped_mask = df["drug_class"].isna()
    n_unmapped = int(unmapped_mask.sum())
    if n_unmapped:
        unmapped_raw_values = df.loc[unmapped_mask, "drug_class_raw"].value_counts()
        logger.warning(f"{n_unmapped} entries matched no canonical drug family and were "
                        f"KEPT with drug_class=None (not dropped). Top unmapped raw values:\n"
                        f"{unmapped_raw_values.head(20).to_string()}")
        logger.warning("These rows will be excluded from multiclass training (no valid label) "
                        "but remain in the dataset for the binary ARG/non-ARG task. Consider "
                        "extending CANONICAL_FAMILIES in src/label_harmonization.py if any of "
                        "these are meaningful drug classes worth adding.")

    after_counts = df["drug_class"].value_counts(dropna=False)
    after_n_classes = df["drug_class"].nunique()

    df.to_csv(labels_csv, index=False)

    logger.info(f"Before: {before_n_rows} rows, {before_n_classes} raw drug_class values")
    logger.info(f"After:  {len(df)} rows, {after_n_classes} canonical drug classes")
    logger.info(f"Canonical class distribution:\n{after_counts.to_string()}")
    logger.info(f"Harmonized labels written -> {labels_csv}")
    logger.info("Next: re-run prepare_dataset.py -> extract_features.py -> train_models.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize drug_class labels in positive_labels.csv")
    parser.add_argument("--labels-csv", default="data/training/positive/positive_labels.csv")
    args = parser.parse_args()

    main(Path(args.labels_csv))
