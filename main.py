"""
AMR-PREDICT: Command-line pipeline controller.

Runs the full flow described in Section 25 of the project plan:
QC -> gene prediction -> ARG similarity screening -> feature extraction
-> ML prediction -> decision engine -> report.

Usage:
    python main.py --genome data/reference/genome.fasta --outdir results/run1
"""
import argparse
import pandas as pd
from pathlib import Path

from src.qc import compute_genome_qc
from src.gene_prediction import predict_genes
from src.similarity_screening import run_arg_screening, parse_screening_results
from src.sequence_features import extract_features_from_fasta as extract_dna_features
from src.protein_features import extract_features_from_fasta as extract_protein_features
from src.prediction import load_models, predict_genes as run_ml_prediction
from src.decision_engine import final_classification
from src.report_generation import save_csv_report, save_html_report


def run_pipeline(genome_fasta: str, outdir: str) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Genome QC...")
    qc = compute_genome_qc(genome_fasta)
    print(qc)

    print("[2/6] Gene prediction (Prodigal)...")
    genes = predict_genes(genome_fasta, outdir / "genes")

    print("[3/6] ARG similarity screening (ABRicate/CARD)...")
    screen_csv = run_arg_screening(genes["genes_nt_fasta"], outdir / "screening.tsv")
    screening_hits = {h["gene"]: h for h in parse_screening_results(screen_csv)}

    print("[4/6] Feature extraction...")
    dna_feats = extract_dna_features(genes["genes_nt_fasta"])
    protein_feats = extract_protein_features(genes["genes_aa_fasta"])
    features = dna_feats.join(protein_feats, how="inner")

    print("[5/6] ML prediction + explainability...")
    models = load_models()
    ml_results = run_ml_prediction(features, models)

    # Scaled feature matrix for downstream SHAP explainability
    X_scaled = pd.DataFrame(
        models["scaler"].transform(features),
        columns=features.columns,
        index=features.index,
    )

    print("[6/6] Hybrid decision engine + report...")
    final_rows = []
    for gene_id, row in ml_results.iterrows():
        sim = screening_hits.get(gene_id, {}).get("result", "Unknown")
        decision = final_classification(sim, row["arg_prediction"], row["arg_probability"])
        final_rows.append({**row.to_dict(), "similarity_result": sim, **decision})

    results_df = pd.DataFrame(final_rows, index=ml_results.index)

    csv_path = save_csv_report(results_df, outdir / "ARG_prediction_results.csv")
    html_path = save_html_report(results_df, qc, outdir / "AMR_prediction_report.html")

    print(f"Done. Results: {csv_path}\nReport: {html_path}")

    return {
        "results": results_df,
        "qc": qc,
        "features_raw": features,
        "X_scaled": X_scaled,
        "binary_model": models["binary_model"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMR-PREDICT pipeline")
    parser.add_argument("--genome", required=True, help="Path to input genome FASTA")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()
    output = run_pipeline(args.genome, args.outdir)
