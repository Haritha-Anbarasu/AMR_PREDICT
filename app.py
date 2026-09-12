"""
AMR-PREDICT Streamlit App

Run with: streamlit run app.py

Pages: Home -> Genome Quality -> ARG Screening -> Results Table
       -> Explainability -> Download Report
"""
import os
import streamlit as st
import pandas as pd
from pathlib import Path

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
DNA_BANNER_SVG = """
<div style="text-align:center; margin-bottom: 1rem;">
<svg width="100%" height="140" viewBox="0 0 900 140" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="strandA" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#2563eb"/>
      <stop offset="100%" stop-color="#7c3aed"/>
    </linearGradient>
    <linearGradient id="strandB" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="100%" stop-color="#0891b2"/>
    </linearGradient>
  </defs>
  <path d="M0,20 C 75,90 150,-30 225,20 C 300,90 375,-30 450,20 C 525,90 600,-30 675,20 C 750,90 825,-30 900,20"
        fill="none" stroke="url(#strandA)" stroke-width="6" stroke-linecap="round"/>
  <path d="M0,110 C 75,40 150,160 225,110 C 300,40 375,160 450,110 C 525,40 600,160 675,110 C 750,40 825,160 900,110"
        fill="none" stroke="url(#strandB)" stroke-width="6" stroke-linecap="round"/>
  <g stroke="#94a3b8" stroke-width="2">
    <line x1="37"  y1="55"  x2="37"  y2="75"/>
    <line x1="112" y1="65"  x2="112" y2="90"/>
    <line x1="187" y1="45"  x2="187" y2="70"/>
    <line x1="262" y1="55"  x2="262" y2="75"/>
    <line x1="337" y1="65"  x2="337" y2="90"/>
    <line x1="412" y1="45"  x2="412" y2="70"/>
    <line x1="487" y1="55"  x2="487" y2="75"/>
    <line x1="562" y1="65"  x2="562" y2="90"/>
    <line x1="637" y1="45"  x2="637" y2="70"/>
    <line x1="712" y1="55"  x2="712" y2="75"/>
    <line x1="787" y1="65"  x2="787" y2="90"/>
    <line x1="862" y1="45"  x2="862" y2="70"/>
  </g>
</svg>
</div>
"""

if page == "Home":
    st.markdown(DNA_BANNER_SVG, unsafe_allow_html=True)

    st.markdown(
        """
        # 🧬 AMR-PREDICT
        ### Machine Learning-Based Antibiotic Resistance Gene Prediction

        **A computational platform for sequence-based prediction of antibiotic resistance genes**

        Antimicrobial resistance (AMR) is a major global health concern, driven in part by the
        acquisition and dissemination of **antibiotic resistance genes (ARGs)**. Rapid identification
        of these genetic determinants is essential for understanding resistance mechanisms and
        supporting AMR research.

        **AMR-PREDICT** is a machine learning-based bioinformatics application developed to predict
        potential **antibiotic resistance genes from biological sequence data**. The platform
        integrates sequence processing, feature extraction, and machine learning to provide a rapid
        computational approach for ARG prediction.
        """
    )

    st.markdown("### 🔬 Analytical Workflow")
    st.markdown(
        "**Sequence Input** → **Sequence Processing** → **Feature Extraction** "
        "→ **Machine Learning Prediction** → **ARG Classification & Results**"
    )

    st.markdown("### Key Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "- Sequence-based ARG prediction\n"
            "- Machine learning-driven classification\n"
            "- Automated computational analysis"
        )
    with col2:
        st.markdown(
            "- Rapid screening of sequence data\n"
            "- User-friendly prediction interface\n"
            "- Results supporting downstream AMR research"
        )

    st.markdown("### Research Application")
    st.markdown(
        "AMR-PREDICT can facilitate the **preliminary computational screening and characterization "
        "of potential antibiotic resistance determinants**, providing researchers with a scalable "
        "approach for investigating AMR-associated sequences."
    )

    st.markdown(
        "> **Integrating machine learning and bioinformatics for computational "
        "antimicrobial resistance research.**"
    )

    st.warning(
        "**Research Use Only:** AMR-PREDICT provides computational predictions that should be "
        "experimentally validated before biological or clinical interpretation."
    )

    st.divider()

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
