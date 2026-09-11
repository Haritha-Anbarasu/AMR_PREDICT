"""
Runs the trained binary + multiclass models on new (unclassified) genes
and returns predictions with probabilities.
"""
import joblib
import pandas as pd
from pathlib import Path


def load_models(model_dir: str = "models") -> dict:
    model_dir = Path(model_dir)
    return {
        "binary_model": joblib.load(model_dir / "binary_model.pkl"),
        "scaler": joblib.load(model_dir / "scaler.pkl"),
        "multiclass_model": joblib.load(model_dir / "multiclass_model.pkl"),
        "label_encoder": joblib.load(model_dir / "label_encoder.pkl"),
    }


def predict_genes(X: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Run binary ARG prediction on all genes, then multiclass prediction
    only on genes predicted as ARG. Returns a results DataFrame indexed
    by gene_id.
    """
    X_scaled = models["scaler"].transform(X)

    binary_pred = models["binary_model"].predict(X_scaled)
    binary_proba = models["binary_model"].predict_proba(X_scaled)[:, 1]

    results = pd.DataFrame({
        "arg_prediction": ["ARG" if p == 1 else "Non-ARG" for p in binary_pred],
        "arg_probability": binary_proba.round(4),
    }, index=X.index)

    arg_mask = results["arg_prediction"] == "ARG"
    if arg_mask.any():
        X_arg_scaled = X_scaled[arg_mask.values]
        class_pred = models["multiclass_model"].predict(X_arg_scaled)
        class_labels = models["label_encoder"].inverse_transform(class_pred)
        results.loc[arg_mask, "antibiotic_class"] = class_labels
    else:
        results["antibiotic_class"] = None

    results["confidence"] = results["arg_probability"].apply(
        lambda p: "High" if p >= 0.90 else "Moderate" if p >= 0.70 else "Low"
    )

    return results


if __name__ == "__main__":
    print("Import this module: load_models() then predict_genes(feature_df, models)")
