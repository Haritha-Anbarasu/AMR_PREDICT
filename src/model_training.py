"""
Modules 6-7: Machine Learning Model Training

Trains:
  1. Binary model  -> ARG vs Non-ARG
  2. Multiclass model -> antibiotic resistance class (trained only on ARG-labeled genes)

Uses a cluster-based train/test split (see cluster_split.py) rather than
a random split, to avoid near-duplicate sequence leakage between train
and test sets.

All configurable parameters (random seed, split ratio, CV folds,
hyperparameter search strategy) are read from config.yaml via
src/config.py rather than hardcoded — see that file for the rationale.
After training, a full performance report (metrics, confusion matrix,
classification report, feature importance, reproducibility info) is
generated automatically via src/model_report.py.

IMPORTANT — parallelism: base estimators (RandomForest/XGBoost/LightGBM)
are deliberately built with n_jobs=1. Parallelism happens ONLY at the
RandomizedSearchCV/GridSearchCV level (n_jobs=-1). Setting n_jobs=-1 on
BOTH the estimator and the search wrapper causes nested parallelism —
the search spawns one worker process per CPU core, and each of those
workers then ALSO tries to claim every core for its own model training.
The result is far more competing threads than physical cores, and the
CPU spends most of its time context-switching rather than computing —
in practice this can turn a 20-30 minute search into many hours on a
mid-range machine (e.g. Ryzen 5 / 16GB). This is a common, easy-to-miss
sklearn pitfall worth stating explicitly rather than fixing silently.
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from src.config import cfg
from src.utils import get_logger, timed_step, ModelError
from src.model_report import generate_performance_report, compare_models_report

logger = get_logger(__name__)

RANDOM_SEED = cfg("random_seed", 42)

# Compact, laptop-friendly hyperparameter grids. Deliberately narrow —
# a full grid across all of these for 3 models under RandomizedSearchCV's
# default n_iter is already a meaningful compute budget on a Ryzen 5 /
# 16GB machine within a two-week project timeline (see config.yaml
# comments for the grid-vs-randomized-search tradeoff rationale).
PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "n_estimators": [200, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
    },
    "lightgbm": {
        "n_estimators": [200, 300, 500],
        "max_depth": [-1, 6, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
    },
}


def _build_base_model(model_key: str):
    # n_jobs=1 on every base estimator — see module docstring. Parallelism
    # is applied once, at the RandomizedSearchCV/GridSearchCV level only.
    if model_key == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1)
    if model_key == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=RANDOM_SEED, n_jobs=1, eval_metric="logloss")
    if model_key == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=RANDOM_SEED, n_jobs=1, verbosity=-1)
    raise ModelError(f"Unknown model key: {model_key}. Expected one of {list(PARAM_GRIDS)}")


@timed_step("Hyperparameter tuning")
def _tune_model(model_key: str, X_train, y_train):
    """Runs RandomizedSearchCV or GridSearchCV per config.yaml settings.
    Parallelism (n_jobs=-1) is applied HERE ONLY — see module docstring
    for why the base estimator must stay single-threaded."""
    base_model = _build_base_model(model_key)
    param_grid = PARAM_GRIDS[model_key]
    search_type = cfg("model.hyperparameter_search", "randomized")
    cv_folds = cfg("model.cv_folds", 5)

    if search_type == "grid":
        search = GridSearchCV(base_model, param_grid, cv=cv_folds, scoring="f1_weighted",
                               n_jobs=-1, verbose=1)
    else:
        n_iter = cfg("model.n_iter_randomized_search", 25)
        search = RandomizedSearchCV(base_model, param_grid, n_iter=n_iter, cv=cv_folds,
                                     scoring="f1_weighted", random_state=RANDOM_SEED,
                                     n_jobs=-1, verbose=1)

    search.fit(X_train, y_train)
    logger.info(f"{model_key}: best params = {search.best_params_} "
                f"(best CV f1_weighted = {search.best_score_:.4f})")
    return search.best_estimator_


def _cluster_aware_split(X_scaled, y, cluster_ids, test_size=0.2):
    if cluster_ids is not None:
        unique_clusters = cluster_ids.unique()
        train_clusters, test_clusters = train_test_split(
            unique_clusters, test_size=test_size, random_state=RANDOM_SEED
        )
        train_mask = cluster_ids.isin(train_clusters)
        return (X_scaled[train_mask.values], X_scaled[~train_mask.values],
                y[train_mask.values] if hasattr(y, "values") else y[train_mask],
                y[~train_mask.values] if hasattr(y, "values") else y[~train_mask])
    return train_test_split(X_scaled, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)


@timed_step("Train binary ARG/non-ARG models")
def train_binary_models(X: pd.DataFrame, y_binary: pd.Series, cluster_ids: pd.Series = None,
                         model_dir: str = None, models_to_train: list = None) -> dict:
    """
    Trains every model listed in config.yaml's model.models_to_train
    (defaults to Random Forest, XGBoost, LightGBM) for the ARG vs
    non-ARG binary task, tunes each with RandomizedSearchCV/GridSearchCV,
    generates a full performance report per model, and returns the
    best-performing model by test F1-score alongside a comparison table.

    If cluster_ids is provided, splits by cluster (all members of a
    cluster go entirely to train or entirely to test) instead of randomly.
    """
    model_dir = Path(model_dir or cfg("paths.models_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    models_to_train = models_to_train or cfg("model.models_to_train", ["random_forest", "xgboost"])
    if not cfg("model.lightgbm_enabled", True) and "lightgbm" in models_to_train:
        models_to_train = [m for m in models_to_train if m != "lightgbm"]

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = _cluster_aware_split(X_scaled, y_binary, cluster_ids)

    all_metrics = []
    trained_models = {}
    for model_key in models_to_train:
        logger.info(f"--- Training {model_key} (binary) ---")
        try:
            clf = _tune_model(model_key, X_train, y_train)
        except ImportError as e:
            logger.warning(f"Skipping {model_key}: {e}")
            continue

        metrics = generate_performance_report(
            model=clf, model_name=f"{model_key}_binary",
            X_test=X_test, y_test=y_test, feature_names=feature_names,
            class_names=["Non-ARG", "ARG"],
        )
        all_metrics.append(metrics)
        trained_models[model_key] = clf
        joblib.dump(clf, model_dir / f"{model_key}_binary_model.pkl")

    if not all_metrics:
        raise ModelError("No binary models were successfully trained — check the errors above.")

    compare_models_report(all_metrics)
    best_model_key = max(all_metrics, key=lambda m: m["f1_score"])["model_name"].replace("_binary", "")
    best_model = trained_models[best_model_key]

    joblib.dump(best_model, model_dir / "binary_model.pkl")   # kept for backward compatibility
    joblib.dump(scaler, model_dir / "scaler.pkl")
    logger.info(f"Best binary model: {best_model_key} — saved as models/binary_model.pkl")

    return {"models": trained_models, "best_model": best_model, "best_model_key": best_model_key,
            "scaler": scaler, "metrics": all_metrics}


@timed_step("Train multiclass antibiotic-class models")
def train_multiclass_models(X: pd.DataFrame, y_class: pd.Series, cluster_ids: pd.Series = None,
                             model_dir: str = None, models_to_train: list = None) -> dict:
    """
    Same multi-model + tuning + reporting flow as train_binary_models(),
    for the antibiotic-class classification task. Only call this on the
    subset of genes already labeled as ARG (X, y_class should exclude
    Non-ARG rows).
    """
    model_dir = Path(model_dir or cfg("paths.models_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    models_to_train = models_to_train or cfg("model.models_to_train", ["random_forest", "xgboost"])
    if not cfg("model.lightgbm_enabled", True) and "lightgbm" in models_to_train:
        models_to_train = [m for m in models_to_train if m != "lightgbm"]

    feature_names = X.columns.tolist()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_class)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = _cluster_aware_split(X_scaled, y_encoded, cluster_ids)

    all_metrics = []
    trained_models = {}
    for model_key in models_to_train:
        logger.info(f"--- Training {model_key} (multiclass) ---")
        try:
            clf = _tune_model(model_key, X_train, y_train)
        except ImportError as e:
            logger.warning(f"Skipping {model_key}: {e}")
            continue

        metrics = generate_performance_report(
            model=clf, model_name=f"{model_key}_multiclass",
            X_test=X_test, y_test=y_test, feature_names=feature_names,
            class_names=label_encoder.classes_.tolist(),
        )
        all_metrics.append(metrics)
        trained_models[model_key] = clf
        joblib.dump(clf, model_dir / f"{model_key}_multiclass_model.pkl")

    if not all_metrics:
        raise ModelError("No multiclass models were successfully trained — check the errors above.")

    compare_models_report(all_metrics)
    best_model_key = max(all_metrics, key=lambda m: m["f1_score"])["model_name"].replace("_multiclass", "")
    best_model = trained_models[best_model_key]

    joblib.dump(best_model, model_dir / "multiclass_model.pkl")   # kept for backward compatibility
    joblib.dump(label_encoder, model_dir / "label_encoder.pkl")
    logger.info(f"Best multiclass model: {best_model_key} — saved as models/multiclass_model.pkl")

    return {"models": trained_models, "best_model": best_model, "best_model_key": best_model_key,
            "label_encoder": label_encoder, "metrics": all_metrics}


# Backward-compatible aliases matching the original single-model function
# names, in case earlier notebooks/scripts already call these directly.
def train_binary_model(*args, **kwargs):
    return train_binary_models(*args, **kwargs)


def train_multiclass_model(*args, **kwargs):
    return train_multiclass_models(*args, **kwargs)


if __name__ == "__main__":
    print("Import this module and call train_binary_models()/train_multiclass_models() "
          "with your feature matrix and labels. Model list, tuning strategy, CV folds, "
          "and random seed are all read from config.yaml.")
