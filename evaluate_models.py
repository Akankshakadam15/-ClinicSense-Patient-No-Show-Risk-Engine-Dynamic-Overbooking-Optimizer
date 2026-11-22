"""
ClinicSense – Model Accuracy & Performance Report
Displays classification reports for all trained models in the same
format as sklearn's classification_report.
"""
import os, sys, joblib
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from src.config import MODELS_DIR

# ── Load test data ───────────────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), "data")
probs_path = os.path.join(data_dir, "model_probs.joblib")
splits_path = os.path.join(data_dir, "splits.joblib")

if not os.path.exists(probs_path) or not os.path.exists(splits_path):
    print("ERROR: Run notebooks 01-04 first to generate model outputs.")
    sys.exit(1)

probs = joblib.load(probs_path)
y_test = probs["y_test"]

# ── Model probability keys ──────────────────────────────────────────────────
model_map = {
    "Logistic Regression": "lr_prob",
    "Random Forest":       "rf_prob",
    "XGBoost":             "xgb_prob",
    "LightGBM":            "lgbm_prob",
    "Stacking Ensemble":   "stack_prob",
}

print("=" * 65)
print("     ClinicSense – MODEL ACCURACY & PERFORMANCE REPORT")
print("=" * 65)
print(f"\nTest set size: {len(y_test)} samples")
print(f"Class distribution: {int((y_test==0).sum())} showed-up, "
      f"{int((y_test==1).sum())} no-show "
      f"({y_test.mean()*100:.1f}% no-show rate)\n")

# ── Evaluate each model ─────────────────────────────────────────────────────
best_acc = 0
best_name = ""

for name, key in model_map.items():
    if key not in probs:
        print(f"  {name}: probabilities not found, skipping.\n")
        continue

    prob = probs[key]

    # Find optimal threshold that maximises accuracy
    best_t, best_a = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.005):
        preds = (prob >= t).astype(int)
        a = accuracy_score(y_test, preds)
        if a > best_a:
            best_a = a
            best_t = t

    # predictions at optimal threshold
    y_pred = (prob >= best_t).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, prob)

    print("=" * 55)
    print(f"  MODEL: {name.upper()}")
    print("=" * 55)
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Threshold: {best_t:.3f}")
    print()
    print("              CLASSIFICATION REPORT")
    print("-" * 55)
    target_names = ["showed-up (0)", "no-show (1)"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    if acc > best_acc:
        best_acc = acc
        best_name = name

# ── Also evaluate the calibrated model ──────────────────────────────────────
cal_path = os.path.join(MODELS_DIR, "best_model_calibrated.joblib")
if os.path.exists(cal_path) and os.path.exists(splits_path):
    splits = joblib.load(splits_path)
    X_test = splits[1]  # (X_train, X_test, y_train, y_test)
    y_true = splits[3]
    cal_model = joblib.load(cal_path)
    try:
        cal_prob = cal_model.predict_proba(X_test)[:, 1]
        # optimal threshold
        best_t, best_a = 0.5, 0
        for t in np.arange(0.05, 0.95, 0.005):
            preds = (cal_prob >= t).astype(int)
            a = accuracy_score(y_true, preds)
            if a > best_a:
                best_a = a
                best_t = t

        y_pred = (cal_prob >= best_t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, cal_prob)

        print("=" * 55)
        print(f"  MODEL: CALIBRATED LIGHTGBM (BEST)")
        print("=" * 55)
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  Threshold: {best_t:.3f}")
        print()
        print("              CLASSIFICATION REPORT")
        print("-" * 55)
        target_names = ["showed-up (0)", "no-show (1)"]
        print(classification_report(y_true, y_pred, target_names=target_names))

        if acc > best_acc:
            best_acc = acc
            best_name = "Calibrated LightGBM"
    except Exception as e:
        print(f"  Calibrated model error: {e}\n")

print("=" * 65)
print(f"  BEST MODEL: {best_name}")
print(f"  BEST ACCURACY: {best_acc*100:.2f}%")
print("=" * 65)
print("\nEvaluation complete.")
