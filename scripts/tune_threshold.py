"""
Threshold tuning for the binary ARG/non-ARG XGBoost model.

Objective: maximize recall subject to precision >= PRECISION_FLOOR,
tuned on the validation set, then evaluated once on the test set.

Run from the AMR_PREDICT project root:
    python tune_threshold.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

PRECISION_FLOOR = 0.85
MODEL_PATH = "models/binary_model.pkl"

# ---------------------------------------------------------------------
# 1. Load model + data
#
# Adjust these lines to match how train_models.py loads its splits.
# Expecting: X_val, y_val, X_test, y_test as pandas/numpy arrays,
# with y in {0: non-ARG, 1: ARG}.
# ---------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

# TODO: replace with your actual data-loading call, e.g.:
# from src.data_loading import load_binary_splits
# X_train, X_val, X_test, y_train, y_val, y_test = load_binary_splits()

X_val, y_val = None, None      # <-- fill in
X_test, y_test = None, None    # <-- fill in

if X_val is None:
    raise RuntimeError(
        "Fill in X_val/y_val/X_test/y_test using your project's data loader "
        "before running this script."
    )

# ---------------------------------------------------------------------
# 2. Sweep thresholds on validation set
# ---------------------------------------------------------------------
val_proba = model.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
# precision_recall_curve returns len(thresholds) = len(precisions) - 1
precisions, recalls = precisions[:-1], recalls[:-1]

valid = precisions >= PRECISION_FLOOR
if not valid.any():
    print(f"No threshold on the val set reaches precision >= {PRECISION_FLOOR}.")
    print(f"Best achievable precision: {precisions.max():.4f}")
    best_idx = np.argmax(precisions)
else:
    # among thresholds meeting the precision floor, pick the one with max recall
    candidate_recalls = np.where(valid, recalls, -1)
    best_idx = np.argmax(candidate_recalls)

best_threshold = thresholds[best_idx]
print(f"Chosen threshold: {best_threshold:.4f}")
print(f"Val precision={precisions[best_idx]:.4f}  recall={recalls[best_idx]:.4f}")

# ---------------------------------------------------------------------
# 3. Apply the tuned threshold to the test set (single, final check)
# ---------------------------------------------------------------------
test_proba = model.predict_proba(X_test)[:, 1]
test_pred_default = (test_proba >= 0.5).astype(int)
test_pred_tuned = (test_proba >= best_threshold).astype(int)

def report(y_true, y_pred, label):
    print(f"\n--- {label} ---")
    print(f"accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"precision: {precision_score(y_true, y_pred):.4f}")
    print(f"recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"f1       : {f1_score(y_true, y_pred):.4f}")
    print("confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(y_true, y_pred))

report(y_test, test_pred_default, "Default threshold (0.5)")
report(y_test, test_pred_tuned, f"Tuned threshold ({best_threshold:.4f})")

# ---------------------------------------------------------------------
# 4. Alternative / complementary: class-weighted retraining
#
# If threshold tuning alone doesn't buy enough recall, retrain XGBoost
# with scale_pos_weight = (n_negative / n_positive) on the TRAIN split,
# then repeat the threshold sweep above on the new model.
#
# Example:
#   from xgboost import XGBClassifier
#   n_neg = (y_train == 0).sum()
#   n_pos = (y_train == 1).sum()
#   weighted_model = XGBClassifier(
#       **best_params,   # reuse the tuned hyperparameters from train_models.py
#       scale_pos_weight=n_neg / n_pos,
#       random_state=42,
#   )
#   weighted_model.fit(X_train, y_train)
# ---------------------------------------------------------------------
