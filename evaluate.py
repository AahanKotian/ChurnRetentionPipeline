"""
evaluate.py
-----------
Evaluate the trained churn model.
Includes threshold tuning via precision-recall curve.
Run after train.py.
"""

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)
import os, json

ARTIFACT_DIR = "artifacts"
os.makedirs("outputs", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────
pipeline = joblib.load(f"{ARTIFACT_DIR}/churn_model.pkl")
X_test   = np.load(f"{ARTIFACT_DIR}/X_test.npy")
y_test   = np.load(f"{ARTIFACT_DIR}/y_test.npy")

# ── Predictions ───────────────────────────────────────────────
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ── Core metrics (default threshold = 0.5) ───────────────────
auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC : {auc:.3f}\n")
print("── Default threshold (0.50) ──")
print(classification_report(y_test, y_pred,
      target_names=["retained", "churned"]))

# ── Confusion matrix ──────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["retained", "churned"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion matrix — default threshold (0.50)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_default.png", dpi=150)
plt.close()

# ── Threshold tuning ──────────────────────────────────────────
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thresholds, precision_vals[:-1], label="precision", color="#1D9E75", linewidth=2)
ax.plot(thresholds, recall_vals[:-1],    label="recall",    color="#185FA5", linewidth=2)
f1_vals = 2 * precision_vals[:-1] * recall_vals[:-1] / (
    precision_vals[:-1] + recall_vals[:-1] + 1e-9)
ax.plot(thresholds, f1_vals, label="F1", color="#7F77DD", linewidth=2, linestyle="--")
ax.axvline(0.35, color="#888780", linestyle=":", linewidth=1.5, label="candidate (0.35)")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision · Recall · F1 vs classification threshold")
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("outputs/precision_recall_curve.png", dpi=150)
plt.close()
print("Plots saved → outputs/")

# ── Apply tuned threshold ─────────────────────────────────────
THRESHOLD = 0.35   # ← tune this based on your precision-recall curve
y_pred_tuned = (y_proba >= THRESHOLD).astype(int)
print(f"\n── Tuned threshold ({THRESHOLD}) ──")
print(classification_report(y_test, y_pred_tuned,
      target_names=["retained", "churned"]))

# ── Save summary ──────────────────────────────────────────────
summary = {
    "roc_auc":          round(float(auc), 4),
    "default_threshold": 0.50,
    "tuned_threshold":   THRESHOLD,
}
with open("outputs/eval_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Summary saved → outputs/eval_summary.json")
