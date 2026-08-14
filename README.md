<h1 align="center">🌫️ Pearls AQI Predictor</h1>

<p align="center">
  <b>An end-to-end, 100% serverless ML system that forecasts the Air Quality Index (AQI) for Lahore, Pakistan — 3 days ahead.</b>
</p>

<p align="center">
  <!-- Badges become live once GitHub Actions + Streamlit deploy are set up -->
  <img alt="Feature Pipeline" src="https://img.shields.io/badge/feature%20pipeline-hourly-2ea44f">
  <img alt="Training Pipeline" src="https://img.shields.io/badge/training%20pipeline-daily-2ea44f">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

<p align="center">
  <a href="#"><b>🔗 Live Dashboard</b></a> · <a href="docs/report/">📄 Project Report</a>
</p>

---

## Overview

Lahore is consistently one of the most polluted cities in the world. This project builds a
production-style, fully automated pipeline that predicts its air quality for the next 3 days,
so residents can plan ahead and vulnerable groups get early warnings.

The system runs entirely on free, serverless infrastructure:

```
 Open-Meteo API ──▶ Feature Pipeline ──▶ Hopsworks Feature Store ──▶ Training Pipeline ──▶ Model Registry
   (weather +         (hourly, via                                     (daily, via              │
   pollutants)       GitHub Actions)                                 GitHub Actions)            ▼
                                                                                        Streamlit Dashboard
                                                                                        (live 3-day forecast)
```

## Architecture

| Stage | Tool | What it does |
|-------|------|--------------|
| **1. Data source** | Open-Meteo (Air Quality + Weather) | Free, keyless historical & forecast data |
| **2. Feature pipeline** | Python + GitHub Actions (hourly) | Fetch raw data → engineer features → store |
| **3. Feature Store** | Hopsworks (free tier) | Central store for features & targets |
| **4. Training pipeline** | scikit-learn, XGBoost, TensorFlow (daily) | Train, evaluate, register best model |
| **5. Model Registry** | Hopsworks | Versioned models + metrics |
| **6. Dashboard** | Streamlit Community Cloud | Live + forecasted AQI, SHAP explanations, alerts |

## Tech stack

`Python 3.11` · `pandas` · `scikit-learn` · `XGBoost` · `TensorFlow` · `Hopsworks` ·
`GitHub Actions` · `SHAP` · `Streamlit` · `Plotly`

## Project structure

```
.
├── .github/workflows/       # CI/CD: hourly feature + daily training pipelines
├── src/aqi/                 # Installable Python package
│   ├── config.py            # Central settings (city, Hopsworks, forecast horizon)
│   ├── data/                # API fetching + AQI computation
│   ├── features/            # Feature engineering
│   ├── pipelines/           # feature / backfill / training / inference entrypoints
│   ├── models/              # Model definitions & training helpers
│   └── utils/               # Logging, shared helpers
├── notebooks/               # EDA & experimentation
├── app/                     # Streamlit dashboard
├── tests/                   # Unit tests
├── docs/report/             # Final project report
├── requirements*.txt        # Split deps (core / app / deep-learning / dev)
└── pyproject.toml
```

## Quickstart (local)

```bash
# 1. Create the environment (Python 3.11)
conda create -n aqi python=3.11 -y
conda activate aqi

# 2. Install the package + dependencies
pip install -e .
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env      # then fill in HOPSWORKS_API_KEY

# 4. Run the feature pipeline once
python -m aqi.pipelines.feature_pipeline
```

## Results (so far)

- **Dataset:** 696,960 hourly rows · 22 cities · Jan 2023 → Aug 2026.
- **Best model:** global **XGBoost** — RMSE **20.6**, MAE **11.8**, **R² 0.82**,
  beating a persistence baseline by **26%** (walk-forward validation).
- **Coverage:** 3-day (72-hour) forecasts with 80% prediction intervals for every
  city; predicts for *any* location on demand.

See [docs/RUNNING.md](docs/RUNNING.md) to run it end-to-end, and
[docs/interview-prep.md](docs/interview-prep.md) for the full write-up.

## Roadmap / progress

- [x] **Module 1** — Feature pipeline (fetch + engineer 72 features)
- [x] **Multi-city** — 22 cities across all provinces; single global model
- [x] **Module 2** — Historical backfill (697k rows)
- [x] **Module 3** — Exploratory Data Analysis (figures + findings)
- [x] **Module 4** — Training pipeline (baseline / Ridge / RF / XGBoost + intervals)
- [ ] **Module 5** — Deep learning model (LSTM / TensorFlow) *(next)*
- [x] **Module 6** — SHAP explainability
- [x] **Module 7** — React + FastAPI dashboard (map, toggle, intervals) *(local; deploy pending)*
- [x] **Module 8** — Hazardous-AQI alerts
- [x] **CI/CD** — Hourly feature + daily training workflows *(ready; needs repo + secrets)*
- [ ] **Storage** — Switch local Parquet → Hopsworks *(needs API key)*
- [ ] **Deploy** — Frontend + backend to the cloud
- [ ] **Final** — LaTeX report & documentation

---

<p align="center"><i>Built for the 10Pearls Data Science Internship.</i></p>
