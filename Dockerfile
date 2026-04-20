FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (maximises Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source and model artifact
COPY app.py .
COPY artifacts/churn_model.pkl artifacts/churn_model.pkl

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
