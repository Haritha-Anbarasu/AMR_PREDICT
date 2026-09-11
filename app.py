"""
AMR-PREDICT CLI Pipeline Controller

This script acts as the central execution bridge for Module 1 through Module 8.
It resolves systemic PATH conflicts for sandboxed environment executions.
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# ==================== ABRICATE ENVIRONMENT ENFORCEMENT ====================
# 
abricate_bin_path = os.path.abspath("abricate-master/bin")
if abricate_bin_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.path.pathsep + abricate_bin_path

# ஒருவேளை சிஸ்டம் PATH மாறினாலும் நேரடியாக பைனரியை இயக்குவதற்கான உலகளாவிய பாத்
if os.path.exists(os.path.join(abricate_bin_path, "abricate")):
    os.environ["ABRICATE_EXEC_PATH"] = os.path.join(abricate_bin_path, "abricate")
# ==========================================================================

from src.qc import run_genome_qc
from src.gene_prediction import predict_genes
from src.similarity_screening import run_similarity_screening
from src.sequence_features import extract_sequence_features
from src.protein_features import extract_protein_features
from src.prediction import run_ml_predictions
from src.decision_engine import apply_hybrid_decision_logic
from src.explainability import generate_shap_explainability
from src.report_generation import export_pipeline_reports

def run_pipeline(genome_fasta_path, output_dir):
    """
    Runs the entire AMR-PREDICT machine learning screening workflow.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/8] Running Genome Quality Control on: {genome_fasta_path}...")
    qc_metrics = run_genome_qc(genome_fasta_path)
    
    print("[2/8] Predicting genes using Prodigal wrapper...")
    predicted_genes_fasta = predict_genes(genome_fasta_path, out_path / "genes")
    
    print("[3/8] Running sequence similarity screening using ABRicate...")
    # இங்கு ஏப்ரிகேட்டின் சரியான பாத் பயன்படுத்தப்படுவதை உறுதி செய்கிறது
    similarity_results_tsv = run_similarity_screening(predicted_genes_fasta, out_path / "similarity")
    
    print("[4/8] Extracting DNA and k-mer sequence features...")
    dna_features = extract_sequence_features(predicted_genes_fasta)
    
    print("[5/8] Extracting protein physicochemical features...")
    protein_features = extract_protein_features(predicted_genes_fasta)
    
    print("[6/8] Executing machine learning model predictions...")
    ml_outputs = run_ml_predictions(dna_features, protein_features)
    
    print("[7/8] Applying hybrid similarity + ML decision engine logic...")
    final_results_df = apply_hybrid_decision_logic(similarity_results_tsv, ml_outputs)
    
    print("[8/8] Generating SHAP explainability matrices for high-confidence predictions...")
    generate_shap_explainability(ml_outputs, out_path / "explainability")
    
    print("Exporting compilation reports...")
    export_pipeline_reports(final_results_df, qc_metrics, out_path)
    
    return {
        "results": final_results_df,
        "qc": qc_metrics,
        "X_scaled": ml_outputs.get("X_scaled"),
        "binary_model": ml_outputs.get("binary_model")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMR-PREDICT Whole-Genome Screening Pipeline")
    parser.add_argument("--genome", required=True, help="Path to input bacterial genome (FASTA format)")
    parser.add_argument("--outdir", required=True, help="Directory to save execution output logs and files")
    
    args = parser.parse_args()
    run_pipeline(args.genome, args.outdir)
    print("Pipeline process completed successfully.")
