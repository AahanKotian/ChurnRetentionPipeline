# Churn Prediction Pipeline

End-to-end customer churn prediction: SQL feature engineering → scikit-learn model → FastAPI REST endpoint → Docker deployment.

## Project structure

```
churn-prediction/
├── sql/
│   ├── 01_extract_customers.sql   # Pull raw 90-day event data
│   └── 02_feature_engineering.sql # Compute behavioral features
├── preprocess.py                  # Clean, encode, scale, split
├── train.py                       # Train RandomForest + save .pkl
├── evaluate.py                    # Metrics, threshold tuning, plots
├── app.py                         # FastAPI prediction server
├── Dockerfile                     # Container for deployment
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Install dependencies
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare data
Run the SQL queries against your database and export the result to:
```
data/customer_features.csv
```

### 3. Run the pipeline
```bash
python preprocess.py   # → artifacts/X_train.npy, scaler.pkl, etc.
python train.py        # → artifacts/churn_model.pkl
python evaluate.py     # → outputs/precision_recall_curve.png, eval_summary.json
```

### 4. Start the API locally
```bash
uvicorn app:app --reload
# Swagger UI → http://localhost:8000/docs
```

### 5. Test a prediction
```bash
curl -X POST "http://localhost:8000/predict?customer_id=cust_001" \
  -H "Content-Type: application/json" \
  -d '{
    "days_since_active": 21,
    "weekly_event_rate": 1.4,
    "tenure_days": 180,
    "days_to_renewal": 14,
    "feature_breadth": 0.12,
    "total_session_mins": 320,
    "plan_encoded": 1
  }'
```

Expected response:
```json
{
  "customer_id": "cust_001",
  "churn_prob": 0.712,
  "churn_flag": true,
  "risk_tier": "high"
}
```

### 6. Docker
```bash
# Build and run
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Threshold tuning

The default classification threshold is **0.35** (lower than sklearn's 0.5 default) to prioritise recall — catching more churners at the cost of some false alarms.

Adjust in `app.py`:
```python
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.35"))
```

Or set at runtime:
```bash
CHURN_THRESHOLD=0.45 uvicorn app:app
```

See `evaluate.py` for the full precision-recall curve analysis.

## Deployment

| Platform | Command |
|----------|---------|
| Railway / Render | Connect GitHub repo — auto-detects Dockerfile |
| Google Cloud Run | `gcloud run deploy churn-api --image gcr.io/$PROJECT/churn-api --platform managed` |
| AWS App Runner | Push image to ECR → deploy via App Runner console |

## Model performance (example)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.87 |
| Recall (churn, threshold=0.35) | ~83% |
| Precision | ~79% |
| F1 | ~81% |

*Replace with your actual numbers after running `evaluate.py`.*
