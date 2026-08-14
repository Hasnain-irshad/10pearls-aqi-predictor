# Running the Pearls AQI Predictor locally

End-to-end, on your machine. All commands assume the `aqi` conda env is active
(`conda activate aqi`) and you're in the project root.

## 0. One-time setup
```bash
conda create -n aqi python=3.11 -y
conda activate aqi
pip install -e . -r requirements.txt         # core pipeline deps
pip install -r requirements-api.txt          # FastAPI backend
```
(Optional) copy `.env.example` to `.env` and add your `HOPSWORKS_API_KEY` — the
storage layer then writes to Hopsworks instead of local Parquet automatically.

## 1. Data → features
```bash
# One-off: build the historical dataset for all 22 cities (~10-15 min).
python -m aqi.pipelines.backfill --start 2023-01-01

# Hourly job (also refreshes latest data):
python -m aqi.pipelines.feature_pipeline --past-days 7
```
Writes `data/processed/features.parquet` (local) or to Hopsworks (if keyed).

## 2. Explore (EDA)
```bash
python -m aqi.eda        # -> docs/images/*.png + docs/eda_findings.md
```

## 3. Train the model
```bash
python -m aqi.pipelines.training_pipeline
# -> models_local/aqi_model.joblib + docs/model_metrics.md (RMSE/MAE/R² table)
```

## 4. Explain (SHAP)
```bash
python -m aqi.models.explain   # -> docs/images/shap_importance.png
```

## 5. Forecast (batch inference)
```bash
python -m aqi.pipelines.inference   # -> data/processed/predictions.json
```

## 6. Serve the API
```bash
uvicorn aqi.api.main:app --reload --port 8000
# http://localhost:8000/api/predictions  ·  /api/cities  ·  /docs (Swagger)
```

## 7. Run the dashboard
```bash
cd web
npm install       # first time only
npm run dev       # http://localhost:5173
```
The dashboard reads from `http://localhost:8000` by default. To point at a
deployed backend, set `VITE_API_URL` before `npm run build`.

## Typical order
backfill → eda → training → explain → inference → uvicorn → npm run dev
