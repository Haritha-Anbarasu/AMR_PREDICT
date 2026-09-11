"""
AMR-PREDICT Streamlit App

Run with: streamlit run app.py

Pages: Home -> Genome Quality -> ARG Screening -> Results Table
       -> Explainability -> Download Report
"""
import os
import subprocess
import streamlit as st
import pandas as pd
from pathlib import Path

# --- ABRicate Autoinstall Setup (Streamlit Deployment Fix) ---
@st.cache_resource
def install_bioinformatics_tools():
    # Only handles ABRicate local download and path registration (cd-hit and sudo safely removed)
    if subprocess.run("which abricate", shell=True, capture_output=True).returncode != 0:
        if not os.path.exists("abricate-master"):
            os.system("curl -L -s https://github.com -o master.zip")
            os.system("unzip -q master.zip && rm master.zip")
        
        abricate_bin_path = os.path.abspath("abricate-master/bin")
        if abricate_bin_path not in os.environ["PATH"]:
            os.environ["PATH"] += os.path.pathsep + abricate_bin_path

# Execute safe initialization steps at startup
install_bioinformatics_tools()
# ----------------------------------------------------------------------

from main import run_pipeline
from src.streamlit_explainability import render_explainability_page

st.set_page_config(page_title="AMR-PREDICT", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = None
if "qc" not in st.session_state:
    st.session_state.qc = None
if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None
if "binary_model" not in st.session_state:
    st.session_state.binary_model = None

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Genome Quality", "ARG Screening", "Results Table", "Explainability", "Download Report"],
)

# ---------------- Home ----------------
if page == "Home":
    st.title("AMR-PREDICT")
    st.subheader("Machine Learning-Based Antibiotic Resistance Gene Prediction")

    uploaded = st.file_uploader("Upload bacterial genome (FASTA)", type=["fasta", "fna", "fa"])

    if uploaded and st.button("Start Analysis"):
        tmp_path = Path("results/uploaded_genome.fasta")
        tmp_path.parent.mkdir(exist_ok=True)
        tmp_path.write_bytes(uploaded.getvalue())

        with st.spinner("Running pipeline — this can take a few minutes..."):
            pipeline_output = run_pipeline(str(tmp_path), "results/streamlit_run")
            st.session_state.results = pipeline_output["results"]
            st.session_state.qc = pipeline_output["qc"]
            st.session_state.X_scaled = pipeline_output["X_scaled"]
            st.session_state.binary_model = pipeline_output["binary_model"]

        st.success("Analysis complete. Use the sidebar to explore results.")

# ---------------- Genome Quality ----------------
elif page == "Genome Quality":
    st.header("Genome Quality Control")
    if st.session_state.qc:
        qc = st.session_state.qc
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Genome Size", f"{qc['genome_size_mb']} Mb")
        col2.metric("GC Content", f"{qc['gc_content_pct']}%")
        col3.metric("Contigs", qc["num_contigs"])
        col4.metric("Quality", qc["quality_status"])
    else:
        st.info("Run an analysis from the Home page first.")

# ---------------- ARG Screening ----------------
elif page == "ARG Screening":
    st.header("ARG Screening Summary")
    df = st.session_state.results
    if df is not None:
        st.metric("Total Genes Screened", len(df))
        st.metric("Confirmed ARGs", (df["final_category"] == "Confirmed ARG").sum())
        st.metric("Potential ARGs", (df["final_category"] == "Potential ARG").sum())
        st.metric("Non-ARG-like", (df["final_category"] == "Non-ARG-like sequence").sum())
    else:
        st.info("Run an analysis from the Home page first.")

# ---------------- Results Table ----------------
elif page == "Results Table":
    st.header("ARG Prediction Results")
    df = st.session_state.results
    if df is not None:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run an analysis from the Home page first.")

# ---------------- Explainability ----------------
elif page == "Explainability":
    render_explainability_page(
        model=st.session_state.binary_model,
        X_scaled=st.session_state.X_scaled,
        results_df=st.session_state.results,
        model_name="binary_best",
    )

# ---------------- Download Report ----------------
elif page == "Download Report":
    st.header("Download Report")
    df = st.session_state.results
    if df is not None:
        st.download_button("Download CSV", df.to_csv().encode(), "ARG_prediction_results.csv")
    else:
        st.info("Run an analysis from the Home page first.")
