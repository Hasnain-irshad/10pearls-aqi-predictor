# Backend image for the Pearls AQI Predictor FastAPI API.
# Slim: serves committed forecasts/eval/monitoring/leaderboard JSON and loads the
# model from the committed bundle (models_local/) — no Feature Store connection
# needed to serve traffic.
FROM python:3.11-slim

WORKDIR /app

# libgomp1 = the OpenMP runtime XGBoost needs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

# App code + committed artifacts (predictions/eval/monitoring/leaderboard + model).
COPY . .
# Editable install so PROJECT_ROOT (parents[2] of aqi/config.py) resolves to /app,
# where the committed data/ and models_local/ live.
RUN pip install --no-cache-dir -e . --no-deps

ENV PORT=8000
CMD ["sh", "-c", "uvicorn aqi.api.main:app --host 0.0.0.0 --port ${PORT}"]
