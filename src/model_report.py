"""
src/model_report.py

Generates a comprehensive model evaluation report after training,
covering everything expected in a PG-level dissertation Results
section: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion
matrix, full classification report, and feature importance — saved as
both a human-readable HTML report and a machine-readable CSV, plus a
reproducibility statement (software versions + confirmed random seed)
so the exact conditions that produced the numbers are recorded
alongside them.

Usage:
    from src.model_report import generate_performance_report

    generate_performance_report(
        model=clf, model_name="random_forest_binary",
        X_test=X_test, y_test=y_test, feature_names=X.columns.tolist(),
    )
"""
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — safe to call from scripts, notebooks, or Streamlit
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)

from src.config import cfg
from src.utils import get_logger, timed_step
from src.environment_info import get_environment_info

logger = get_logger(__name__)


def _reports_dir() -> Path:
    d = Path(cfg("paths.reports_dir", "reports")) / "model_performance"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot_confusion_matrix(y_test, y_pred, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def _plot_feature_importance(model, feature_names, out_path: Path, top_n: int = 20):
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.3)))
    ax.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#1F3864")
    ax.set_xlabel("Feature Importance (Gini / Gain)")
    ax.set_title(f"Top {top_n} Features")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return dict(zip([feature_names[i] for i in order], importances[order].tolist()))


@timed_step("Generate model performance report")
def generate_performance_report(model, model_name: str, X_test, y_test, feature_names: list,
                                 class_names: list = None) -> dict:
    """
    Evaluates a trained model on a held-out test set and writes:
        reports/model_performance/{model_name}_metrics.csv
        reports/model_performance/{model_name}_report.html
        reports/model_performance/{model_name}_confusion_matrix.png
        reports/model_performance/{model_name}_feature_importance.png

    Returns the computed metrics dict (also useful for cross-model
    comparison tables — see src/model_training.py compare_models()).
    """
    out_dir = _reports_dir()
    y_test = np.asarray(y_test)
    y_pred = model.predict(X_test)

    is_binary = len(np.unique(y_test)) == 2
    average = "binary" if is_binary else "weighted"

    metrics = {
        "model_name": model_name,
        "n_test_samples": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
    }

    # ROC-AUC / PR-AUC require predicted probabilities
    roc_auc, pr_auc = None, None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        try:
            if is_binary:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                pr_auc = average_precision_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                pr_auc = average_precision_score(
                    pd.get_dummies(y_test).values, y_proba, average="weighted"
                )
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC/PR-AUC for {model_name}: {e}")

    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc

    class_report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    class_report_text = classification_report(y_test, y_pred, zero_division=0)

    display_labels = class_names if class_names else sorted(np.unique(y_test).tolist())
    cm_path = out_dir / f"{model_name}_confusion_matrix.png"
    cm = _plot_confusion_matrix(y_test, y_pred, display_labels, cm_path)

    fi_path = out_dir / f"{model_name}_feature_importance.png"
    feature_importance = _plot_feature_importance(model, feature_names, fi_path)

    # ---- CSV (machine-readable) ----
    metrics_csv_path = out_dir / f"{model_name}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)

    # ---- HTML (human-readable, dissertation-ready) ----
    html_path = out_dir / f"{model_name}_report.html"
    _write_html_report(
        html_path, model_name, metrics, class_report_text, cm_path, fi_path,
        random_seed=cfg("random_seed", 42),
    )

    logger.info(f"{model_name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}, "
                f"roc_auc={roc_auc if roc_auc is None else round(roc_auc, 4)}")
    logger.info(f"Report written -> {html_path}")

    return metrics


def _write_html_report(html_path, model_name, metrics, class_report_text, cm_path, fi_path, random_seed):
    env_info = get_environment_info()
    env_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in env_info.items())

    metric_rows = "".join(
        f"<tr><td>{k}</td><td>{'%.4f' % v if isinstance(v, float) else v}</td></tr>"
        for k, v in metrics.items() if k != "model_name"
    )

    fi_img_tag = (f'<img src="{fi_path.name}" style="max-width:700px;">'
                  if fi_path and fi_path.exists() else "<p>Model has no feature_importances_ attribute.</p>")

    html = f"""
    <html><head><title>Model Performance Report: {model_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 950px; margin: 40px auto; color: #222; }}
        h1 {{ color: #1F3864; }}
        h2 {{ color: #B8860B; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
        th {{ background: #1F3864; color: white; }}
        pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }}
        .note {{ color: #555; font-size: 0.9em; }}
    </style></head>
    <body>
        <h1>Model Performance Report — {model_name}</h1>
        <p class="note">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
        Random seed: {random_seed} (fixed across data split and model training for reproducibility)</p>

        <h2>Summary Metrics</h2>
        <table><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>

        <h2>Confusion Matrix</h2>
        <img src="{cm_path.name}" style="max-width:500px;">

        <h2>Classification Report</h2>
        <pre>{class_report_text}</pre>

        <h2>Feature Importance</h2>
        {fi_img_tag}

        <h2>Reproducibility — Software Environment</h2>
        <table><tr><th>Component</th><th>Version</th></tr>{env_rows}</table>
    </body></html>
    """
    html_path.write_text(html)


def compare_models_report(all_metrics: list) -> Path:
    """
    Given a list of metrics dicts (one per model, as returned by
    generate_performance_report), writes a single side-by-side
    comparison CSV — the table typically used directly in a
    dissertation Results section.
    """
    out_dir = _reports_dir()
    df = pd.DataFrame(all_metrics).set_index("model_name")
    out_path = out_dir / "model_comparison.csv"
    df.to_csv(out_path)
    logger.info(f"Model comparison table -> {out_path}\n{df.to_string()}")
    return out_path
