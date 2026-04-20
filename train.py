"""
train.py
--------
Train a Random Forest churn classifier and save the pipeline artifact.
Run after preprocess.py.
"""

import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

FEATURES = [
    "days_since_active",
    "weekly_event_rate",
    "tenure_days",
    "days_to_renewal",
    "feature_breadth",
    "total_session_mins",
    "plan_encoded",
]

# ── Load preprocessed splits ──────────────────────────────────
X_train = np.load(f"{ARTIFACT_DIR}/X_train.npy")
y_train = np.load(f"{ARTIFACT_DIR}/y_train.npy")
print(f"Training on {len(X_train):,} samples")

# ── Build pipeline ────────────────────────────────────────────
# Scaler is already applied in preprocess.py, but we bundle a
# pass-through here so the saved pipeline accepts raw features
# at inference time if you skip the preprocess step.
pipeline = Pipeline([
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",   # handles class imbalance
        random_state=42,
        n_jobs=-1,
    ))
])

# ── Train ─────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)
print("Training complete")

# ── Feature importances ───────────────────────────────────────
importances = pipeline["clf"].feature_importances_
feat_imp = dict(zip(FEATURES, importances.round(4)))
feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: -x[1]))
print("\nFeature importances:")
for feat, imp in feat_imp_sorted.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

with open(f"{ARTIFACT_DIR}/feature_importances.json", "w") as f:
    json.dump(feat_imp_sorted, f, indent=2)

# ── Save model ────────────────────────────────────────────────
joblib.dump(pipeline, f"{ARTIFACT_DIR}/churn_model.pkl")
print(f"\nModel saved → {ARTIFACT_DIR}/churn_model.pkl")
