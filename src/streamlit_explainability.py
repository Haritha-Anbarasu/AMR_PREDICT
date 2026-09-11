"""
src/streamlit_explainability.py

Renders the SHAP explainability page inside the Streamlit app. Kept in
its own module (rather than inside src/explainability.py) so that
explainability.py — used by CLI scripts like generate_explanations.py
and by main.py's pipeline — doesn't require Streamlit as a dependency
just to run a report generation script with no UI involved.

Call render_explainability_page() from app.py's Explainability page,
passing:
    model      -> the trained binary model (session_state.binary_model)
    X_scaled   -> the scaled feature DataFrame for this genome's genes,
                  indexed by gene_id (session_state.X_scaled)
    results_df -> the final results table (session_state.results),
                  used only to show prediction/confidence context
                  alongside the SHAP explanation, not required for SHAP
                  itself
    model_name -> label used in generated plot filenames
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.explainability import (
    get_shap_explainer, explain_prediction, waterfall_plot,
    global_summary_plot, global_beeswarm_plot,
)


def render_explainability_page(model, X_scaled: pd.DataFrame, results_df: pd.DataFrame = None,
                                model_name: str = "binary_best"):
    st.header("Prediction Explainability (SHAP)")

    if model is None or X_scaled is None or len(X_scaled) == 0:
        st.info("Run an analysis from the Home page first.")
        return

    tab1, tab2 = st.tabs(["Per-Gene Explanation", "Global Feature Importance"])

    # ---------------- Per-gene (local) explanation ----------------
    with tab1:
        gene_id = st.selectbox("Select a gene", X_scaled.index.tolist())
        X_row = X_scaled.loc[[gene_id]]

        if results_df is not None and gene_id in results_df.index:
            row = results_df.loc[gene_id]
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", row.get("final_category", row.get("arg_prediction", "N/A")))
            probability = row.get("arg_probability", None)
            col2.metric("Probability", f"{probability:.3f}" if probability is not None else "N/A")
            col3.metric("Confidence", row.get("confidence", "N/A"))

        explainer = get_shap_explainer(model)
        contributions = explain_prediction(explainer, X_row, top_n=10)

        st.subheader("Top contributing features")
        st.caption("Positive SHAP values push the prediction toward ARG; "
                    "negative values push toward Non-ARG.")
        contrib_df = pd.DataFrame(contributions).set_index("feature")
        st.dataframe(contrib_df, use_container_width=True)

        st.subheader("Waterfall plot")
        with st.spinner("Generating explanation for this gene..."):
            waterfall_path = waterfall_plot(model, X_row, gene_id, model_name)
        st.image(str(waterfall_path))

    # ---------------- Global explanation ----------------
    with tab2:
        st.write("Feature importance computed across all genes in this genome's analysis. "
                 "This can take a moment for large genomes since SHAP values are computed "
                 "individually for every gene.")
        if st.button("Generate global explanation plots"):
            with st.spinner(f"Computing SHAP values across {len(X_scaled)} genes..."):
                summary_path = global_summary_plot(model, X_scaled, model_name)
                beeswarm_path = global_beeswarm_plot(model, X_scaled, model_name)
            st.subheader("Feature importance (magnitude)")
            st.image(str(summary_path))
            st.subheader("Feature importance (magnitude + direction)")
            st.image(str(beeswarm_path))
