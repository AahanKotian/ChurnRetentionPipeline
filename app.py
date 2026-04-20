"""
app.py
------
FastAPI REST API for churn prediction.
Run locally:  uvicorn app:app --reload
Swagger docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability from behavioral features.",
    version="1.0.0",
)

# ── Load model once at startup ────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/churn_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")

THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.35"))

FEATURE_ORDER = [
    "days_since_active",
    "weekly_event_rate",
    "tenure_days",
    "days_to_renewal",
    "feature_breadth",
    "total_session_mins",
    "plan_encoded",
]


# ── Schemas ───────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    days_since_active:  float = Field(..., ge=0,  description="Days since last login")
    weekly_event_rate:  float = Field(..., ge=0,  description="Avg events per week (90-day window)")
    tenure_days:        float = Field(..., ge=0,  description="Days since signup")
    days_to_renewal:    float = Field(...,        description="Days until contract renewal (negative = expired)")
    feature_breadth:    float = Field(..., ge=0, le=1, description="Distinct features / total events ratio")
    total_session_mins: float = Field(..., ge=0,  description="Total session minutes in window")
    plan_encoded:       int   = Field(..., ge=0, le=2, description="0=free, 1=pro, 2=enterprise")


class PredictionOut(BaseModel):
    customer_id: str
    churn_prob:  float
    churn_flag:  bool
    risk_tier:   str   # "low" | "medium" | "high"


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "model": MODEL_PATH, "threshold": THRESHOLD}


@app.post("/predict", response_model=PredictionOut, tags=["prediction"])
def predict(customer_id: str, features: CustomerFeatures):
    """Return churn probability and risk tier for a single customer."""
    X = np.array([[getattr(features, f) for f in FEATURE_ORDER]])
    prob = float(model.predict_proba(X)[0, 1])
    tier = "high" if prob > 0.65 else "medium" if prob > 0.35 else "low"
    return PredictionOut(
        customer_id=customer_id,
        churn_prob=round(prob, 4),
        churn_flag=prob >= THRESHOLD,
        risk_tier=tier,
    )


@app.post("/predict/batch", response_model=list[PredictionOut], tags=["prediction"])
def predict_batch(customers: list[dict]):
    """
    Batch prediction endpoint.
    Body: [{"customer_id": "...", "features": {...}}, ...]
    """
    results = []
    for item in customers:
        cid = item.get("customer_id", "unknown")
        try:
            feat = CustomerFeatures(**item["features"])
            results.append(predict(cid, feat))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Error for {cid}: {e}")
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
