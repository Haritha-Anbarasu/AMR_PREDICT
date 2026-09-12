"""
AMR-PREDICT Streamlit App

Run with: streamlit run app.py

Pages: Home -> Genome Quality -> ARG Screening -> Results Table
       -> Explainability -> Download Report
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from main import run_pipeline
from src.streamlit_explainability import render_explainability_page

st.set_page_config(page_title="AMR-PREDICT", layout="wide", page_icon="🧬")

# ---------------- Global theme ----------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #eaf3ff 0%, #dbeafe 45%, #eaf3ff 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #dbeafe 0%, #eaf3ff 100%);
    border-right: 1px solid rgba(15, 23, 42, 0.08);
}
section[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

h1, h2, h3, h4 {
    color: #0f172a !important;
}
p, li, span, label, .stMarkdown {
    color: #334155;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #9333ea);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
}
.stButton > button:hover {
    transform: translateY(-1px) scale(1.02);
    box-shadow: 0 6px 18px rgba(124, 58, 237, 0.5);
}

[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(148, 163, 184, 0.4);
    border-radius: 12px;
    padding: 0.5rem;
}

.hero-wrap {
    position: relative;
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
}
.hero-wrap img {
    width: 100%;
    display: block;
    max-height: 300px;
    object-fit: cover;
    filter: brightness(0.55) saturate(1.15);
}
.hero-text {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 2rem 2.5rem;
}
.hero-text h1 {
    font-size: 2.6rem;
    margin: 0;
    background: linear-gradient(90deg, #60a5fa, #c084fc, #f472b6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}
.hero-text p {
    color: #e2e8f0 !important;
    font-size: 1.1rem;
    margin-top: 0.4rem;
    max-width: 640px;
}

.metric-card {
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-left: 5px solid var(--accent, #2563eb);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
}
.metric-card .m-label {
    font-size: 0.8rem;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.2rem;
}
.metric-card .m-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #0f172a !important;
}

.feature-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    height: 100%;
}
.feature-card, .feature-card * {
    color: #1e293b !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def metric_card(label: str, value, color: str = "#2563eb"):
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color:{color};">
            <div class="m-label">{label}</div>
            <div class="m-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1576086213369-97a306d36557"
    "?fm=jpg&q=80&w=1600&auto=format&fit=crop"
)

# ---------------- Session state ----------------
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
    st.markdown(
        f"""
        <div class="hero-wrap">
            <img src="{HERO_IMAGE_URL}" alt="Fluorescence microscopy of DNA">
            <div class="hero-text">
                <h1>🧬 AMR-PREDICT</h1>
                <p>Machine Learning-Based Antibiotic Resistance Gene Prediction</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Image: National Cancer Institute / Unsplash")

    st.markdown(
        """
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
    steps = ["Sequence\nInput", "Sequence\nProcessing", "Feature\nExtraction",
             "ML\nPrediction", "ARG\nClassification"]
    step_colors = ["#3b82f6", "#8b5cf6", "#d946ef", "#f97316", "#22c55e"]
    cols = st.columns(len(steps))
    for c, step, color in zip(cols, steps, step_colors):
        with c:
            st.markdown(
                f"""
                <div style="text-align:center; padding:0.8rem 0.4rem; border-radius:10px;
                            background: #ffffff; border-top: 3px solid {color};
                            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);">
                    <div style="font-weight:600; color:#0f172a; white-space:pre-line;">{step}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Key Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="feature-card">
            ✅ Sequence-based ARG prediction<br>
            ✅ Machine learning-driven classification<br>
            ✅ Automated computational analysis
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="feature-card">
            ⚡ Rapid screening of sequence data<br>
            🖥️ User-friendly prediction interface<br>
            📊 Results supporting downstream AMR research
            </div>
            """,
            unsafe_allow_html=True,
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
    st.header("🧫 Genome Quality Control")
    if st.session_state.qc:
        qc = st.session_state.qc
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Genome Size", f"{qc['genome_size_mb']} Mb", "#3b82f6")
        with col2:
            metric_card("GC Content", f"{qc['gc_content_pct']}%", "#8b5cf6")
        with col3:
            metric_card("Contigs", qc["num_contigs"], "#06b6d4")
        with col4:
            metric_card("Quality", qc["quality_status"], "#22c55e")
    else:
        st.info("Run an analysis from the Home page first.")

# ---------------- ARG Screening ----------------
elif page == "ARG Screening":
    st.header("🧬 ARG Screening Summary")
    df = st.session_state.results
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Total Genes Screened", len(df), "#3b82f6")
        with col2:
            metric_card("Confirmed ARGs", (df["final_category"] == "Confirmed ARG").sum(), "#e11d48")
        with col3:
            metric_card("Potential ARGs", (df["final_category"] == "Potential ARG").sum(), "#f59e0b")
        with col4:
            metric_card("Non-ARG-like", (df["final_category"] == "Non-ARG-like sequence").sum(), "#22c55e")
    else:
        st.info("Run an analysis from the Home page first.")

# ---------------- Results Table ----------------
elif page == "Results Table":
    st.header("📋 ARG Prediction Results")
    df = st.session_state.results
    if df is not None:
        CATEGORY_COLORS = {
            "Confirmed ARG": "#fecaca",       # light red
            "Potential ARG": "#fde68a",       # light amber
            "Non-ARG-like sequence": "#e2e8f0",  # neutral grey (no real highlight)
        }

        def _highlight_row(row):
            color = CATEGORY_COLORS.get(row.get("final_category"), "")
            return [f"background-color: {color}; color: #0f172a"] * len(row)

        if "final_category" in df.columns:
            styled_df = df.style.apply(_highlight_row, axis=1)
            st.dataframe(styled_df, use_container_width=True)

            st.markdown(
                """
                <div style="display:flex; gap:1.5rem; margin-top:0.6rem; flex-wrap:wrap;">
                    <span><span style="display:inline-block;width:12px;height:12px;
                        background:#fecaca;border-radius:3px;margin-right:6px;"></span>Confirmed ARG</span>
                    <span><span style="display:inline-block;width:12px;height:12px;
                        background:#fde68a;border-radius:3px;margin-right:6px;"></span>Potential ARG</span>
                    <span><span style="display:inline-block;width:12px;height:12px;
                        background:#e2e8f0;border-radius:3px;margin-right:6px;"></span>Non-ARG-like sequence</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
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
    st.header("⬇️ Download Report")
    df = st.session_state.results
    if df is not None:
        st.download_button("Download CSV", df.to_csv().encode(), "ARG_prediction_results.csv")
    else:
        st.info("Run an analysis from the Home page first.")
