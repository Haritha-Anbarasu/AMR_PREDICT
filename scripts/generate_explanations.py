"""
scripts/generate_explanations.py

Runs full SHAP explainability (Module 8) against your best trained
binary model and real test-set genes:
    - Global summary (bar) + beeswarm plots — what the model relies on
      overall
    - Local waterfall + force plots for a handful of example genes —
      why specific predictions were made, including at least one
      correct ARG call, one correct Non-ARG call, and (if present) one
      misclassification, since examining what the model got wrong is
      often more informative for a dissertation Discussion section
      than only showing successes.

Usage:
    python scripts/generate_explanations.py
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, timed_step, ModelError
from src.explainability import (
    get_shap_explainer, global_summary_plot, global_beeswarm_plot,
    explain_prediction, waterfall_plot, force_plot_html,
)

logger = get_logger(__name__)

NON_FEATURE_COLUMNS = {"label", "drug_class", "drug_class_all", "drug_class_raw", "source_db"}


def load_test_features(features_dir: Path):
    path = features_dir / "test_features.csv"
    if not path.exists():
        raise ModelError(f"Expected {path}. Run scripts/extract_features.py first.")
    df = pd.read_csv(path).set_index("gene_id")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df, feature_cols


def pick_example_genes(df: pd.DataFrame, y_pred) -> dict:
    """Selects a small, illustrative set of genes to locally explain:
    a correctly-called ARG, a correctly-called Non-ARG, and a
    misclassification if one exists in the test set."""
    df = df.copy()
    df["_pred"] = y_pred
    df["_true"] = (df["label"] == "ARG").astype(int)

    examples = {}
    correct_arg = df[(df["_true"] == 1) & (df["_pred"] == 1)]
    correct_non_arg = df[(df["_true"] == 0) & (df["_pred"] == 0)]
    misclassified = df[df["_true"] != df["_pred"]]

    if len(correct_arg):
        examples["correct_ARG"] = correct_arg.index[0]
    if len(correct_non_arg):
        examples["correct_Non_ARG"] = correct_non_arg.index[0]
    if len(misclassified):
        examples["misclassified"] = misclassified.index[0]

    return examples


@timed_step("Generate all SHAP explanations")
def main(model_dir: Path, features_dir: Path, model_name: str):
    model_path = model_dir / "binary_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise ModelError(f"Expected {model_path} and {scaler_path}. Run scripts/train_models.py first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df, feature_cols = load_test_features(features_dir)
    X_test_scaled = scaler.transform(df[feature_cols])
    X_test = pd.DataFrame(X_test_scaled, columns=feature_cols, index=df.index)

    logger.info(f"Loaded {len(X_test)} test genes, {len(feature_cols)} features")

    # --- Global explanations ---
    global_summary_plot(model, X_test, model_name)
    global_beeswarm_plot(model, X_test, model_name)

    # --- Local explanations on a small illustrative sample ---
    y_pred = model.predict(X_test_scaled)
    example_genes = pick_example_genes(df, y_pred)

    explainer = get_shap_explainer(model)
    for label, gene_id in example_genes.items():
        logger.info(f"Explaining example '{label}': {gene_id}")
        X_row = X_test.loc[[gene_id]]

        contributions = explain_prediction(explainer, X_row)
        logger.info(f"Top contributing features for {gene_id}:")
        for c in contributions:
            logger.info(f"    {c['feature']:<25} {c['shap_value']:+.4f}  ({c['direction']})")

        waterfall_plot(model, X_row, gene_id, model_name)
        force_plot_html(model, X_row, gene_id, model_name)

    logger.info("All SHAP explanations written to reports/explainability/")
    logger.info(f"Example genes explained: {example_genes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate global and local SHAP explanations")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--model-name", default="binary_best",
                         help="Label used in output filenames (doesn't need to match the algorithm name)")
    args = parser.parse_args()

    main(Path(args.model_dir), Path(args.features_dir), args.model_name)
