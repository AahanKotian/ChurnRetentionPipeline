"""
preprocess.py
-------------
Load SQL output CSV, handle nulls, encode categoricals,
scale features, and produce train/test splits.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ── Config ────────────────────────────────────────────────────
DATA_PATH   = DATA_PATH   = "data/customer_features.csv"
OUTPUT_DIR  = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "days_since_active",
    "weekly_event_rate",
    "tenure_days",
    "days_to_renewal",
    "feature_breadth",
    "total_session_mins",
    "plan_encoded",
]
TARGET = "churned"

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows, {df[TARGET].mean():.1%} churn rate")

# ── Handle missing values ─────────────────────────────────────
df["days_to_renewal"].fillna(-1, inplace=True)   # -1 = no contract
df["feature_breadth"].fillna(0, inplace=True)    # 0 = no events
df["total_session_mins"].fillna(0, inplace=True)

# ── Encode plan_type → integer ────────────────────────────────
le = LabelEncoder()
df["plan_encoded"] = le.fit_transform(df["plan_type"])
joblib.dump(le, f"{OUTPUT_DIR}/label_encoder.pkl")
print(f"Plan classes: {dict(enumerate(le.classes_))}")

# ── Split ─────────────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,   # preserve churn ratio in both splits
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Scale ─────────────────────────────────────────────────────
# NOTE: scaler is fit on TRAIN only, then applied to test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

# ── Persist splits for train.py ───────────────────────────────
import numpy as np
np.save(f"{OUTPUT_DIR}/X_train.npy", X_train_scaled)
np.save(f"{OUTPUT_DIR}/X_test.npy",  X_test_scaled)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train.values)
np.save(f"{OUTPUT_DIR}/y_test.npy",  y_test.values)

print("Preprocessing complete. Artifacts saved to", OUTPUT_DIR)
