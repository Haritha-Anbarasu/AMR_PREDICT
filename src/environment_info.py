"""
src/environment_info.py

Captures the Python version and installed versions of every key package
the pipeline depends on, so a model performance report can state exactly
what software produced it. This is the standard "reproducibility
statement" expected in a dissertation Methods section or a paper's
supplementary material — a reported accuracy figure is not really
reproducible without knowing which scikit-learn/XGBoost/LightGBM
version generated it, since default hyperparameters and algorithm
internals do change between versions.
"""
import sys
import platform
from importlib.metadata import version, PackageNotFoundError

TRACKED_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm",
    "shap", "biopython", "matplotlib", "streamlit", "joblib", "pyyaml",
]


def get_environment_info() -> dict:
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in TRACKED_PACKAGES:
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            info[pkg] = "not installed"
    return info


def environment_info_as_markdown() -> str:
    info = get_environment_info()
    lines = ["| Component | Version |", "|---|---|"]
    for k, v in info.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)
