# Churn Prediction Pipeline

End-to-end customer churn prediction: SQL feature engineering → scikit-learn model → FastAPI REST endpoint → Docker deployment.

**Built and tested on Windows (Python 3.12, CMD).**

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.979** |
| Recall (churn, threshold=0.50) | **99%** (79/80 churners caught) |
| Precision | 74% |
| F1 score | 84% |
| False negatives | **1** missed churner out of 80 |
| False positives | 28 unnecessary alerts |

### Confusion matrix (240 test customers, threshold=0.50)

```
                Predicted retained   Predicted churned
Actually retained       132                  28
Actually churned          1                  79
```

### Feature importances

| Feature | Importance |
|---------|-----------|
| days_since_active | 0.8139 |
| total_session_mins | 0.0522 |
| weekly_event_rate | 0.0435 |
| tenure_days | 0.0299 |
| feature_breadth | 0.0287 |
| days_to_renewal | 0.0233 |
| plan_encoded | 0.0084 |

> **Threshold note:** With ROC-AUC of 0.979 you can raise the threshold to ~0.65 in production — the precision-recall curve peaks there, catching nearly as many churners with far fewer false alarms.

---

## Project structure

```
churn-prediction/
├── sql/
│   ├── 01_extract_customers.sql    # Pull raw 90-day event data
│   └── 02_feature_engineering.sql  # Compute behavioral features
├── data/
│   ├── churn.db                    # SQLite database (1,200 customers, 45k events)
│   └── customer_features.csv       # Pre-generated feature CSV
├── artifacts/
│   ├── churn_model.pkl             # Trained Random Forest pipeline
│   ├── scaler.pkl                  # StandardScaler
│   ├── label_encoder.pkl           # LabelEncoder for plan_type
│   ├── X_train.npy / X_test.npy   # Train/test splits
│   ├── y_train.npy / y_test.npy
│   └── feature_importances.json   # Real importances from training run
├── outputs/
│   ├── precision_recall_curve.png  # Threshold tuning chart
│   ├── confusion_matrix_default.png
│   └── eval_summary.json
├── preprocess.py                   # Clean, encode, scale, split
├── train.py                        # Train model + save .pkl
├── evaluate.py                     # Metrics, threshold tuning, plots
├── run_sql.py                      # Run SQL files via DuckDB → export CSV
├── app.py                          # FastAPI prediction server
├── Dockerfile
├── requirements.txt
└── index.html                      # GitHub Pages project site
```

---

## Quickstart (Windows)

### 1. Extract the project
```
unzip churn-prediction.zip
cd churn-prediction
```

### 2. Create and activate virtual environment
```bat
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install dependencies
```bat
pip install -r requirements.txt
```

### 4. Run the pipeline
Always `cd` into the project folder first — scripts use relative paths.

```bat
cd C:\path\to\churn-prediction

python preprocess.py
python train.py
python evaluate.py
```

### 5. Start the API
```bat
uvicorn app:app --port 8080
```

Open **http://localhost:8080/docs** for the interactive Swagger UI.

### 6. Test a prediction
```bat
curl -X POST "http://localhost:8080/predict?customer_id=cust_0001" ^
  -H "Content-Type: application/json" ^
  -d "{\"days_since_active\":21,\"weekly_event_rate\":1.4,\"tenure_days\":180,\"days_to_renewal\":14,\"feature_breadth\":0.12,\"total_session_mins\":320,\"plan_encoded\":1}"
```

---

## Common Windows gotchas

**`FileNotFoundError: data/customer_features.csv`**
You're running the script from the wrong directory. Do `cd` into the project folder first, then run `python preprocess.py`.

**`SyntaxError: (unicode error) 'unicodeescape'`**
Don't edit file paths in the scripts to use Windows backslashes (`C:\Users\...`). The original `"data/customer_features.csv"` with forward slashes works fine on Windows. Don't change it.

**`WinError 10013` on uvicorn**
Port 8000 is blocked. Use `uvicorn app:app --port 8080` instead.

**`'import' is not recognized`**
You typed Python code into CMD. Type `python` first to open the Python shell (`>>>` prompt), then enter your code.

---

## Regenerate features from SQL

Only needed if you modify the SQL files:

```bat
pip install duckdb
python run_sql.py
```

This loads `data/churn.db` into DuckDB, runs both SQL files, and re-exports `data/customer_features.csv`.

---

## Docker

```bat
docker build -t churn-api .
docker run -p 8080:8000 churn-api
```

---

## Deploy to Railway (free)

1. Push repo to GitHub (see below)
2. Force-add the model artifact first:
   ```bat
   git add -f artifacts/churn_model.pkl artifacts/scaler.pkl artifacts/label_encoder.pkl
   git commit -m "Add model artifacts"
   git push
   ```
3. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub → select this repo
4. Railway detects the Dockerfile automatically — live in ~2 minutes

---

## API reference

### `GET /`
Health check.

### `POST /predict?customer_id={id}`

**Request body:**
```json
{
  "days_since_active":  21,
  "weekly_event_rate":  1.4,
  "tenure_days":        180,
  "days_to_renewal":    14,
  "feature_breadth":    0.12,
  "total_session_mins": 320,
  "plan_encoded":       1
}
```
`plan_encoded`: 0 = free, 1 = pro, 2 = enterprise

**Response:**
```json
{
  "customer_id": "cust_0001",
  "churn_prob":  0.241,
  "churn_flag":  false,
  "risk_tier":   "low"
}
```
`risk_tier`: `"low"` (< 0.35) · `"medium"` (0.35–0.65) · `"high"` (> 0.65)

### `POST /predict/batch`
Same as above but accepts a list of customers.

---

## Tech stack

| Layer | Tool |
|-------|------|
| Database | SQLite + DuckDB |
| Feature engineering | SQL (DATE_DIFF, CTEs) |
| Data processing | Pandas, NumPy |
| ML | scikit-learn (RandomForestClassifier, Pipeline) |
| API | FastAPI + Pydantic |
| Server | Uvicorn |
| Container | Docker |
| Cloud | Railway / Google Cloud Run |

---
