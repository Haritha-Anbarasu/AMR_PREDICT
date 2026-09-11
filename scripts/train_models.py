"""
scripts/train_models.py

Trains the binary (ARG vs Non-ARG) and multiclass (antibiotic drug
class) models using the pre-made cluster-aware train/val/test feature
splits from extract_features.py — NOT a fresh internal split. This
matters: the train/val/test boundary was already fixed at the sequence-
cluster level in prepare_dataset.py, and re-splitting here would
silently reintroduce the leakage risk that whole pipeline stage exists
to prevent.

Flow per model (binary and, separately, multiclass):
    train split  -> RandomizedSearchCV/GridSearchCV hyperparameter
                     tuning (internal CV happens within train only)
    test split   -> final held-out performance report (HTML + CSV)
    val split    -> reported alongside test as a secondary check

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --skip-multiclass
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger, timed_step, ModelError
from src.config import cfg
from src.model_training import _tune_model, RANDOM_SEED
from src.model_report import generate_performance_report, compare_models_report

logger = get_logger(__name__)

NON_FEATURE_COLUMNS = {"label", "drug_class", "source_db"}


def load_features(features_dir: Path, split_name: str) -> pd.DataFrame:
    path = features_dir / f"{split_name}_features.csv"
    if not path.exists():
        raise ModelError(f"Expected {path}. Run scripts/extract_features.py first.")
    return pd.read_csv(path).set_index("gene_id")


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


@timed_step("Train binary ARG/non-ARG models")
def train_binary(features_dir: Path, model_dir: Path, models_to_train: list) -> dict:
    train_df = load_features(features_dir, "train")
    val_df = load_features(features_dir, "val")
    test_df = load_features(features_dir, "test")

    feature_cols = get_feature_columns(train_df)
    logger.info(f"Binary task: {len(feature_cols)} features, "
                f"{len(train_df)} train / {len(val_df)} val / {len(test_df)} test sequences")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = (train_df["label"] == "ARG").astype(int)
    y_val = (val_df["label"] == "ARG").astype(int)
    y_test = (test_df["label"] == "ARG").astype(int)

    all_metrics = []
    trained_models = {}
    for model_key in models_to_train:
        logger.info(f"--- Training {model_key} (binary) ---")
        try:
            clf = _tune_model(model_key, X_train, y_train)
        except ImportError as e:
            logger.warning(f"Skipping {model_key}: {e} (not installed)")
            continue

        val_metrics = generate_performance_report(
            model=clf, model_name=f"{model_key}_binary_val",
            X_test=X_val, y_test=y_val, feature_names=feature_cols,
            class_names=["Non-ARG", "ARG"],
        )
        logger.info(f"{model_key} val check: accuracy={val_metrics['accuracy']:.4f}, "
                    f"f1={val_metrics['f1_score']:.4f}")

        test_metrics = generate_performance_report(
            model=clf, model_name=f"{model_key}_binary",
            X_test=X_test, y_test=y_test, feature_names=feature_cols,
            class_names=["Non-ARG", "ARG"],
        )
        all_metrics.append(test_metrics)
        trained_models[model_key] = clf
        joblib.dump(clf, model_dir / f"{model_key}_binary_model.pkl")

    if not all_metrics:
        raise ModelError("No binary models were successfully trained — check errors above.")

    compare_models_report(all_metrics)
    best = max(all_metrics, key=lambda m: m["f1_score"])
    best_key = best["model_name"].replace("_binary", "")
    best_model = trained_models[best_key]

    joblib.dump(best_model, model_dir / "binary_model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    logger.info(f"Best binary model: {best_key} (test f1={best['f1_score']:.4f}) "
                f"-> models/binary_model.pkl")

    return {"best_model_key": best_key, "metrics": all_metrics}


@timed_step("Train multiclass antibiotic-class models")
def train_multiclass(features_dir: Path, model_dir: Path, models_to_train: list, min_class_samples: int):
    train_df = load_features(features_dir, "train")
    val_df = load_features(features_dir, "val")
    test_df = load_features(features_dir, "test")

    train_arg = train_df[train_df["label"] == "ARG"].copy()
    val_arg = val_df[val_df["label"] == "ARG"].copy()
    test_arg = test_df[test_df["label"] == "ARG"].copy()

    class_counts = train_arg["drug_class"].value_counts()
    valid_classes = class_counts[class_counts >= min_class_samples].index.tolist()
    dropped_classes = class_counts[class_counts < min_class_samples].index.tolist()
    if dropped_classes:
        logger.warning(f"Dropping {len(dropped_classes)} drug classes with < {min_class_samples} "
                        f"training examples (too few to train reliably): {dropped_classes}")

    train_arg = train_arg[train_arg["drug_class"].isin(valid_classes)]
    val_arg = val_arg[val_arg["drug_class"].isin(valid_classes)]
    test_arg = test_arg[test_arg["drug_class"].isin(valid_classes)]

    if train_arg["drug_class"].nunique() < 2:
        logger.warning("Fewer than 2 usable drug classes after filtering — skipping multiclass training.")
        return None

    feature_cols = get_feature_columns(train_arg)
    logger.info(f"Multiclass task: {len(valid_classes)} classes, {len(feature_cols)} features, "
                f"{len(train_arg)} train / {len(val_arg)} val / {len(test_arg)} test sequences")

    label_encoder = LabelEncoder()
    label_encoder.fit(train_arg["drug_class"])

    def encode_safe(df):
        mask = df["drug_class"].isin(label_encoder.classes_)
        return df[mask], label_encoder.transform(df.loc[mask, "drug_class"])

    train_arg, y_train = encode_safe(train_arg)
    val_arg, y_val = encode_safe(val_arg)
    test_arg, y_test = encode_safe(test_arg)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_arg[feature_cols])
    X_val = scaler.transform(val_arg[feature_cols])
    X_test = scaler.transform(test_arg[feature_cols])

    all_metrics = []
    trained_models = {}
    for model_key in models_to_train:
        logger.info(f"--- Training {model_key} (multiclass) ---")
        try:
            clf = _tune_model(model_key, X_train, y_train)
        except ImportError as e:
            logger.warning(f"Skipping {model_key}: {e} (not installed)")
            continue

        if len(val_arg):
            val_metrics = generate_performance_report(
                model=clf, model_name=f"{model_key}_multiclass_val",
                X_test=X_val, y_test=y_val, feature_names=feature_cols,
                class_names=label_encoder.classes_.tolist(),
            )
            logger.info(f"{model_key} val check: accuracy={val_metrics['accuracy']:.4f}, "
                        f"f1={val_metrics['f1_score']:.4f}")

        test_metrics = generate_performance_report(
            model=clf, model_name=f"{model_key}_multiclass",
            X_test=X_test, y_test=y_test, feature_names=feature_cols,
            class_names=label_encoder.classes_.tolist(),
        )
        all_metrics.append(test_metrics)
        trained_models[model_key] = clf
        joblib.dump(clf, model_dir / f"{model_key}_multiclass_model.pkl")

    if not all_metrics:
        logger.warning("No multiclass models were successfully trained.")
        return None

    compare_models_report(all_metrics)
    best = max(all_metrics, key=lambda m: m["f1_score"])
    best_key = best["model_name"].replace("_multiclass", "")
    best_model = trained_models[best_key]

    joblib.dump(best_model, model_dir / "multiclass_model.pkl")
    joblib.dump(scaler, model_dir / "multiclass_scaler.pkl")
    joblib.dump(label_encoder, model_dir / "label_encoder.pkl")
    logger.info(f"Best multiclass model: {best_key} (test f1={best['f1_score']:.4f}) "
                f"-> models/multiclass_model.pkl")

    return {"best_model_key": best_key, "metrics": all_metrics, "classes": valid_classes}


def main(features_dir: Path, model_dir: Path, skip_multiclass: bool):
    model_dir.mkdir(parents=True, exist_ok=True)

    models_to_train = cfg("model.models_to_train", ["random_forest", "xgboost"])
    if not cfg("model.lightgbm_enabled", True) and "lightgbm" in models_to_train:
        models_to_train = [m for m in models_to_train if m != "lightgbm"]
    logger.info(f"Models to train: {models_to_train} (random_seed={RANDOM_SEED})")

    train_binary(features_dir, model_dir, models_to_train)

    if not skip_multiclass:
        min_class_samples = cfg("model.multiclass_min_class_samples", 15)
        train_multiclass(features_dir, model_dir, models_to_train, min_class_samples)
    else:
        logger.info("Skipping multiclass training (--skip-multiclass)")

    logger.info("Model training complete. See reports/model_performance/ for full results, "
                "reports/model_performance/model_comparison.csv for the side-by-side table, "
                "and models/ for saved .pkl files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary and multiclass ARG prediction models")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--skip-multiclass", action="store_true")
    args = parser.parse_args()

    main(Path(args.features_dir), Path(args.model_dir), args.skip_multiclass)
