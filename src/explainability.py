"""
src/explainability.py

Module 8: Explainable AI (SHAP)

Provides both GLOBAL explanations (which features matter most across
the whole model — summary/importance plots) and LOCAL explanations
(why THIS specific gene was classified as it was — waterfall/force
plots), using SHAP's TreeExplainer, which works directly and exactly
(not approximately) with tree-based models like Random Forest, XGBoost,
and LightGBM — no model-agnostic approximation needed since all three
of your trained models are tree ensembles.

Why both global AND local explanations matter for this project:
global explanations answer "what does the model generally rely on to
call something an ARG?" (useful for validating the model learned
biologically sensible signal — e.g. confirming codon usage bias or
aromaticity actually matter, rather than some spurious artifact).
Local explanations answer "why did the model flag THIS gene?" — the
actual clinically/scientifically relevant question when a user uploads
a real genome and wants to trust (or scrutinize) an individual
prediction, and the whole point of the hybrid decision engine reporting
a confidence level rather than a bare yes/no.

Tested against SHAP 0.52 with a real trained RandomForestClassifier
before shipping: that version returns a 3D array
(n_samples, n_features, n_classes) from TreeExplainer.shap_values() for
RandomForest, rather than the list-of-arrays some SHAP versions/other
model types return. A naive `shap_values[0]` index on that 3D case
silently returns a (features x classes) MATRIX instead of a
features-length VECTOR, which breaks waterfall/force plots without an
obvious error at the call site — the local-explanation helpers below
explicitly normalize all three shapes (list, 3D array, plain 2D array)
so this doesn't depend on which SHAP version or model type is in use.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import cfg
from src.utils import get_logger, timed_step

logger = get_logger(__name__)


def _reports_dir() -> Path:
    d = Path(cfg("paths.reports_dir", "reports")) / "explainability"
    d.mkdir(parents=True, exist_ok=True)
    return d


@timed_step("Build SHAP explainer")
def get_shap_explainer(model):
    """
    Builds a SHAP TreeExplainer for a trained tree-based model
    (RandomForestClassifier, XGBClassifier, or LGBMClassifier — all
    supported natively and exactly, not approximated).
    """
    return shap.TreeExplainer(model)


def _get_class1_shap_values(explainer, X):
    """
    Normalizes SHAP's output across model types/versions into a single
    (n_samples, n_features) array representing the positive-class
    (index 1, i.e. "ARG") contribution for binary tasks, or the raw
    per-class array for multiclass. Different SHAP/sklearn/XGBoost
    version combinations return either a list-of-arrays (one per class)
    or a single 3D array (n_samples, n_features, n_classes) — this
    handles both without the caller needing to know which. Confirmed
    against SHAP 0.52 (RandomForest returns the 3D form; XGBoost/
    LightGBM binary classifiers typically return a plain 2D array).
    """
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return raw[1] if len(raw) > 1 else raw[0]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        return raw[:, :, 1] if raw.shape[2] > 1 else raw[:, :, 0]
    return raw


def _get_single_row_shap(explainer, X_row):
    """
    Same normalization as _get_class1_shap_values, but for a SINGLE row
    (used by the local-explanation plots): returns a 1D (n_features,)
    SHAP value array and a scalar base_value for the positive class,
    regardless of whether the installed SHAP version returns a list,
    a 3D array, or a plain 2D array for this model type. Tested
    directly against RandomForest (3D-array case), which is the shape
    that broke a naive `shap_values[0]` index (it silently returned a
    features x classes MATRIX instead of a features-length VECTOR).
    """
    raw = explainer.shap_values(X_row)
    base = explainer.expected_value

    if isinstance(raw, list):
        values = raw[1][0] if len(raw) > 1 else raw[0][0]
        base_arr = np.atleast_1d(base)
        base_val = base_arr[1] if len(base_arr) > 1 else base_arr[0]
        return np.asarray(values), float(base_val)

    raw = np.asarray(raw)
    if raw.ndim == 3:
        n_classes = raw.shape[2]
        class_idx = 1 if n_classes > 1 else 0
        values = raw[0, :, class_idx]
        base_arr = np.atleast_1d(base)
        base_val = base_arr[class_idx] if len(base_arr) > class_idx else base_arr[0]
        return values, float(base_val)

    values = raw[0]
    base_val = np.atleast_1d(base)[0]
    return values, float(base_val)


# ----------------------------------------------------------------------
# GLOBAL explanations
# ----------------------------------------------------------------------
@timed_step("Generate global SHAP summary plot")
def global_summary_plot(model, X: pd.DataFrame, model_name: str, top_n: int = 20) -> Path:
    """
    Bar-style SHAP summary plot: the top_n features ranked by mean
    absolute SHAP value across the whole dataset — the standard global
    "what matters most to this model" view for a dissertation Results
    section.
    """
    explainer = get_shap_explainer(model)
    shap_values = _get_class1_shap_values(explainer, X)

    out_path = _reports_dir() / f"{model_name}_shap_summary.png"
    plt.figure(figsize=(9, max(5, top_n * 0.35)))
    shap.summary_plot(shap_values, X, max_display=top_n, show=False, plot_type="bar")
    plt.title(f"{model_name} — Global Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Global SHAP summary plot -> {out_path}")
    return out_path


@timed_step("Generate SHAP beeswarm plot")
def global_beeswarm_plot(model, X: pd.DataFrame, model_name: str, top_n: int = 20) -> Path:
    """
    Beeswarm plot: like the bar summary but also shows the DIRECTION
    and spread of each feature's effect (e.g. "high GC content pushes
    toward ARG") rather than just magnitude — richer than the bar plot
    alone and standard practice alongside it.
    """
    explainer = get_shap_explainer(model)
    shap_values = _get_class1_shap_values(explainer, X)

    out_path = _reports_dir() / f"{model_name}_shap_beeswarm.png"
    plt.figure(figsize=(9, max(5, top_n * 0.35)))
    shap.summary_plot(shap_values, X, max_display=top_n, show=False)
    plt.title(f"{model_name} — SHAP Value Distribution (direction + magnitude)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP beeswarm plot -> {out_path}")
    return out_path


# ----------------------------------------------------------------------
# LOCAL explanations (per-gene)
# ----------------------------------------------------------------------
def explain_prediction(explainer, X_row: pd.DataFrame, top_n: int = 10) -> list:
    """
    Returns the top_n features driving ONE specific prediction, as a
    list of {feature, shap_value, direction} dicts — used by the
    Streamlit app to show a per-gene textual explanation without
    needing to render a full plot.
    """
    values, _ = _get_single_row_shap(explainer, X_row)

    contributions = list(zip(X_row.columns, values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            "feature": name,
            "shap_value": round(float(val), 4),
            "direction": "Pushes toward ARG" if val > 0 else "Pushes toward Non-ARG",
        }
        for name, val in contributions[:top_n]
    ]


@timed_step("Generate SHAP waterfall plot for one gene")
def waterfall_plot(model, X_row: pd.DataFrame, gene_id: str, model_name: str,
                    max_display: int = 12) -> Path:
    """
    Waterfall plot for a SINGLE gene: shows exactly how each feature
    pushed the prediction up or down from the model's baseline
    (expected value) to the final output for that one gene — the
    standard "why did the model decide this" visual for one prediction.
    """
    explainer = get_shap_explainer(model)
    values, base_value = _get_single_row_shap(explainer, X_row)

    explanation = shap.Explanation(
        values=values, base_values=base_value,
        data=X_row.iloc[0].values, feature_names=X_row.columns.tolist(),
    )

    safe_gene_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in gene_id)[:80]
    out_path = _reports_dir() / f"{model_name}_waterfall_{safe_gene_id}.png"

    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.title(f"Why this gene was classified: {gene_id}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Waterfall plot for {gene_id} -> {out_path}")
    return out_path


@timed_step("Generate SHAP force plot for one gene")
def force_plot_html(model, X_row: pd.DataFrame, gene_id: str, model_name: str) -> Path:
    """
    Force plot (as a standalone interactive HTML file): the classic
    SHAP "red pushes right, blue pushes left" visual for one gene —
    complements the waterfall plot with an interactive version suitable
    for embedding directly in the Streamlit app.
    """
    explainer = get_shap_explainer(model)
    values, base_value = _get_single_row_shap(explainer, X_row)

    force = shap.force_plot(base_value, values, X_row.iloc[0], feature_names=X_row.columns.tolist())

    safe_gene_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in gene_id)[:80]
    out_path = _reports_dir() / f"{model_name}_force_{safe_gene_id}.html"
    shap.save_html(str(out_path), force)
    logger.info(f"Force plot for {gene_id} -> {out_path}")
    return out_path


if __name__ == "__main__":
    print("Import this module: global_summary_plot(), global_beeswarm_plot() for global "
          "explanations; explain_prediction(), waterfall_plot(), force_plot_html() for "
          "per-gene local explanations. See scripts/generate_explanations.py for a "
          "ready-to-run example over your test set.")
