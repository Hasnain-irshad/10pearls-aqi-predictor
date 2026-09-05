<h1 align="center">🌫️ Pearls AQI Predictor</h1>

<p align="center">
  <b>An end-to-end, effectively serverless ML system that forecasts the Air Quality Index<br>
  up to 72 hours ahead for 22 cities across Pakistan — and keeps itself current, unattended.</b>
</p>

<p align="center">
  <a href="https://10pearlsaqi.me"><b>🔗 Live Dashboard</b></a> ·
  <a href="Thesis/main.pdf"><b>📄 Project Report (PDF, 131 pp)</b></a> ·
  <a href="notebooks/eda.ipynb"><b>📊 EDA Notebook</b></a> ·
  <a href="https://aqi-backend-production-5af4.up.railway.app/api/health"><b>⚙️ API</b></a>
</p>

The dashboard is a **React (Vite)** single-page app, not Streamlit. It was upgraded from
the original Streamlit plan to support the five-tab layout, the interactive map and the
prediction-interval charts.

<p align="center">
  <img alt="Feature pipeline" src="https://img.shields.io/badge/feature%20pipeline-hourly-2ea44f">
  <img alt="Training pipeline" src="https://img.shields.io/badge/training%20pipeline-daily-2ea44f">
  <img alt="Cities" src="https://img.shields.io/badge/cities-22-blue">
  <img alt="Horizon" src="https://img.shields.io/badge/horizon-72h-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

## Live links

| | |
|---|---|
| **Dashboard** | **https://10pearlsaqi.me** |
| **Backend API** | https://aqi-backend-production-5af4.up.railway.app |
| **Project report** | [`Thesis/main.pdf`](Thesis/main.pdf) — 131 pages, 35 figures, 23 tables |
| **EDA notebook** | [`notebooks/eda.ipynb`](notebooks/eda.ipynb) — executed, with outputs |

The dashboard is refreshed by the hourly feature-and-inference workflow and retrained by
the daily training workflow. Its header exposes the timestamp of the forecast document
being served, so stale data is visible rather than hidden.

---

## Overview

Air-quality information in Pakistan is almost entirely a **nowcast**: one number, for one
station, right now — no explanation, no indication of tomorrow. But the decisions that
actually protect health are made *in advance*: whether to keep a child indoors, whether to
move outdoor work, whether an asthmatic should carry a mask.

This project forecasts the **US EPA Air Quality Index up to 72 hours ahead for 22 cities**
Kashmir. More importantly, it is a **system rather than a model**: it ingests fresh data
every hour, retrains every night, refuses to deploy a model that isn't measurably better,
explains every prediction, and monitors its own drift and error — all on free
infrastructure.

### What makes it more than a notebook

| | |
|---|---|
| 🔁 **Keeps itself current** | Hourly ingestion + hourly forecast publication + daily training, on scheduled runners |
| 🏆 **Gated promotion** | A new model ships *only* if it beats the champion's RMSE — a tie is refused |
| 📦 **Durable model registry** | The champion survives the ephemeral runner that trained it |
| 📉 **Honest evaluation** | Per-horizon metrics + 5-fold walk-forward backtest, not one averaged number |
| 🎯 **Uncertainty** | 80% prediction interval per point, widening with lead time |
| 🧠 **Explained** | Global and per-city SHAP attributions rendered as plain English |
| 🎛️ **Interrogable** | What-if simulator: change the drivers, watch the model re-predict |
| 🩺 **Self-monitoring** | PSI drift per feature + forecasts scored against what actually happened |
| 💬 **Grounded advisor** | LLM answers via tool calls into the real forecast (free Gemini/Groq tier) |

---

## Architecture

The system is three **decoupled pipelines** that never call each other — they communicate
only through a feature store and a model registry. That decoupling is what lets the compute
be disposable, and therefore free.

```
                    ┌─────────────────── COMPUTE (GitHub Actions, ephemeral) ───────────────────┐
                    │                                                                            │
 Open-Meteo  ──────▶│  1. Feature pipeline   hourly   fetch → engineer → upsert (22 cities)      │
 (free, keyless)    │  2. Training pipeline  daily    supervised set → 4 candidates → GATE       │
                    │  3. Inference pipeline hourly   champion → 72 h curve + intervals + SHAP   │
                    └────────────────┬──────────────────────────────────┬────────────────────────┘
                                     │ write                            │ read champion
                                     ▼                                  ▲
                    ┌─────────── STATE (Hopsworks, free tier) ──────────┴──────────┐
                    │  Feature Store  aqi_features v2   ~697k rows × 74 cols        │
                    │  Model Registry aqi_global_forecaster (champion = min RMSE)   │
                    └──────────────────────────────┬───────────────────────────────┘
                                                   │ predictions.json committed to this repo
                                                   ▼
              FastAPI backend (Railway) ──────▶ React dashboard (Vercel) ──────▶ 10pearlsaqi.me
                       │
                       └─▶ MCP server + LLM advisor (same grounded tools)
```

**The serving layer reads published documents, never the feature store.** The backend
fetches `predictions.json` from this repository at request time (cached briefly, falling
back to the copy bundled in the image), so the site tracks the pipeline without needing a
redeploy.

---

## Results

Trained on 200,000 multi-horizon rows, split **chronologically** (160,002 train / 39,998
validation) — never randomly.

| Model | RMSE | MAE | R² | |
|---|---:|---:|---:|---|
| **XGBoost** | **19.69** | **12.80** | **0.850** | ← promoted champion |
  │       ├── App.jsx                  5 tabs: Forecast, Analytics & SHAP, Evaluation, Monitoring, What-If
| Ridge regression | 22.85 | 16.01 | 0.797 | linear reference |
| Persistence (baseline) | 25.27 | 15.71 | 0.752 | not eligible to ship |

**22% lower RMSE than persistence.** Walk-forward backtest over 5 rolling folds: **mean RMSE
20.55** (± 3.53) — slightly worse than the single split, which is the honest direction.

### Accuracy by forecast horizon

Error grows with lead time, and this is published rather than hidden:

|---|---:|---:|---:|---:|---:|---:|---:|
| **RMSE** | 6.10 | 7.88 | 10.61 | 14.78 | 21.72 | 27.56 | 27.55 |
| **R²** | 0.985 | 0.975 | 0.957 | 0.917 | 0.819 | 0.708 | 0.695 |
| vs baseline | −34% | +32% | +41% | +35% | +16% | +17% | +20% |

The model beats persistence **at every horizon from 2 hours to 3 days** (by 16–41%).
At **1 hour it does not** — the trivial predictor wins there, because air quality one hour
out is dominated by air quality now. That is reported openly; a single averaged figure would
have concealed it.

### Deep learning was tested, and declined

| Model (predict AQI at +24 h) | RMSE | MAE | R² |
|---|---:|---:|---:|
| LSTM (2 recurrent layers) | **22.12** | 14.32 | 0.799 |
| XGBoost (same task) | 22.63 | 14.48 | 0.789 |
| Persistence | 28.79 | 17.21 | 0.660 |

The LSTM is 2.3% better *at that one horizon* — and was **not promoted**, because it serves
one lead time instead of ten, takes ~25 min to train against ~1 min (which would break the
nightly retrain), and cannot be explained with exact SHAP. Measured, then decided.

---

## Exploratory data analysis

Full analysis with outputs: **[`notebooks/eda.ipynb`](notebooks/eda.ipynb)** ·
summary: [`docs/eda_findings.md`](docs/eda_findings.md) ·
figures: [`docs/images/`](docs/images)

Across **696,960 hourly rows**, 22 cities, Jan 2023 → Aug 2026:

| Finding | What it changed in the design |
|---|---|
| **Faisalabad (156.9) is worse than Lahore (151.5)**; Gilgit cleanest at 75.7 — a 2.07× spread | Justifies multi-city scope; the city that gets all the attention isn't the worst |
| **Evening peak 18:00 (123.1), morning low 09:00 (108.1)** | Cyclical hour encoding; the hourly view helps users find a safer time |
| **No dominant weather driver** (strongest \|r\| = 0.32, surface pressure) | A non-linear model with interactions, not a linear one |
| **Target is right-skewed** (skew 1.74) | Empirical residual quantiles for intervals, not a symmetric Gaussian |
| **Autocorrelation 0.99 @1 h, 0.80 @24 h** | Lag + rolling features; persistence as the baseline to beat |

Regenerate the figures with `python -m aqi.eda`.

---

## Repository structure

```
.
├── README.md                     ← you are here
├── Thesis/                       📄 THE PROJECT REPORT
│   ├── main.pdf                     compiled project report
│   ├── chapters/                    ch1–ch10 LaTeX sources
│   ├── figures/                     diagrams, screenshots and analysis charts
│   └── figures_src/                 diagram sources + render/screenshot scripts
├── notebooks/
│   └── eda.ipynb                 📊 exploratory analysis, executed with outputs
├── src/aqi/                      the Python package
│   ├── config.py                    single source of truth: 22 cities, paths, horizon
│   ├── data/
│   │   ├── openmeteo.py             retrying client for 3 Open-Meteo endpoints
│   │   ├── aqi.py                   US EPA AQI: breakpoints, categories, advice
│   │   ├── store.py                 backend-agnostic facade (Hopsworks | Parquet)
│   │   ├── hopsworks_store.py       feature group / feature view
│   │   └── published.py             reads published artefacts at runtime
│   ├── features/
│   │   ├── engineering.py           21 raw cols → 74 engineered (leakage-controlled)
│   │   └── supervised.py            multi-horizon set + chronological split
│   ├── pipelines/
│   │   ├── feature_pipeline.py      hourly
│   │   ├── backfill.py              one-off history load, resumable
│   │   ├── training_pipeline.py     daily: train → gate → register
│   │   └── inference.py              hourly: 72 h forecast + intervals + SHAP
│   ├── models/
│   │   ├── registry.py              bundle save/load
│   │   ├── leaderboard.py           champion–challenger promotion gate
│   │   ├── evaluate.py              per-horizon + walk-forward backtest
│   │   ├── explain.py               SHAP → plain language
│   │   └── lstm.py                  sequence-model study (not shipped)
│   ├── api/main.py                  FastAPI backend
│   ├── llm.py                       pluggable LLM provider (Gemini | Groq | Claude)
│   ├── advisor.py                   grounded advisor
│   ├── tools.py                     the 4 tools the advisor and MCP both use
│   ├── monitoring.py                PSI drift + realised forecast error
│   ├── whatif.py                    what-if simulator
│   ├── alerts.py                    hazard thresholds
│   └── eda.py                       EDA figures + findings
├── web/                          React (Vite) dashboard
│   └── src/
│       ├── App.jsx                  5 tabs: Forecast, Analytics & SHAP, Evaluation, Monitoring, What-If
│       ├── components/              chart, map, legend, chat, SHAP card, …
│       ├── pages/                   Analytics, ModelEval, Monitoring, WhatIf
│       └── markdown.js              renders advisor replies
├── tests/                        unit tests for AQI, features, statistics and storage
├── docs/                         EDA findings, metrics, leaderboard, monitoring, figures
├── .github/workflows/            3 scheduled pipelines
├── Dockerfile                    slim backend image
└── data/ · models_local/         artefacts (large files git-ignored)
```

---

## Running it locally

```bash
git clone https://github.com/Hasnain-irshad/10pearls-aqi-predictor.git
cd 10pearls-aqi-predictor

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e . -r requirements.txt

cp .env.example .env          # optional: add HOPSWORKS_API_KEY, GEMINI_API_KEY
```

Without a Hopsworks key everything falls back to a local Parquet store, so the whole
pipeline runs offline.

```bash
python -m aqi.pipelines.backfill --start 2024-01-01   # load history (once)
python -m aqi.pipelines.feature_pipeline              # hourly feature-store step
python -m aqi.pipelines.inference                     # hourly forecast publication
python -m aqi.pipelines.training_pipeline             # train + promotion gate
python -m aqi.pipelines.inference                     # write predictions.json
python -m aqi.models.evaluate                         # per-horizon + backtest
python -m aqi.monitoring                              # drift + realised error
python -m aqi.eda                                     # EDA figures

uvicorn aqi.api.main:app --reload --port 8000         # backend
cd web && npm install && npm run dev                  # frontend → localhost:5173

pytest -q                                             # 26 tests
```

> **Windows note:** the `hopsworks` SDK cannot be installed on Windows (a transitive
> dependency fails to build). That's why `data/store.py` is a facade — local development
> uses Parquet, and all feature-store work runs on Linux CI.

---

## The advisor (free tier, no paid key needed)

The chat panel answers from the **real forecast** by calling four declared tools — it never
invents a number. The provider is pluggable:

| Provider | Environment variable | Default model | Cost |
|---|---|---|---|
| **Gemini** | `GEMINI_API_KEY` | `gemini-3.6-flash` | free — [get a key](https://aistudio.google.com/apikey) |
| **Groq** | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | free — [get a key](https://console.groq.com/keys) |
| Claude | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` | paid |

The first key found wins, preferring the free tiers. `GET /api/advisor/models` lists what
your key can actually use.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness + how fresh the served forecast is |
| `GET /api/predictions` · `/{city}` | Full forecast payload, or one city |
| `GET /api/cities` · `/api/categories` | Supported cities; the six EPA bands |
| `GET /api/leaderboard` | Champion–challenger promotion history |
| `GET /api/evaluation` | Per-horizon metrics + walk-forward backtest |
| `GET /api/monitoring` | PSI drift + realised forecast error |
| `GET /api/explain/{city}` | SHAP explanation in plain language |
| `GET /api/predict` | On-demand forecast for **any** coordinate |
| `GET` / `POST /api/whatif` | Scenario simulation |
| `POST /api/chat` | Grounded LLM advisor |

---

## Deployment

| Component | Host | How |
|---|---|---|
| Frontend | Vercel (edge, static) | `cd web && vercel deploy --prod` |
| Backend | Railway (Docker) | `railway up --ci --service aqi-backend` |
| Pipelines | GitHub Actions | hourly features + hourly inference + daily training |
| State | Hopsworks Serverless | free tier |

Running cost is **effectively zero**: ingestion is free and keyless, compute uses free CI
minutes, storage and the frontend are free tiers, and only the small backend container
carries any cost.

---

## Documentation

| Document | |
|---|---|
| [`Thesis/main.pdf`](Thesis/main.pdf) | **Full project report** — 131 pp: introduction, literature review, problem, requirements, design, implementation, UI, testing & evaluation, conclusion |
| [`notebooks/eda.ipynb`](notebooks/eda.ipynb) | Executed exploratory analysis |
| [`docs/eda_findings.md`](docs/eda_findings.md) | EDA summary |
| [`docs/evaluation.md`](docs/evaluation.md) | Per-horizon + walk-forward results |
| [`docs/model_leaderboard.md`](docs/model_leaderboard.md) | Promotion history |
| [`docs/model_metrics.md`](docs/model_metrics.md) | Candidate comparison |
| [`docs/monitoring.md`](docs/monitoring.md) | Drift report |
| [`docs/lstm_metrics.md`](docs/lstm_metrics.md) | Deep-learning comparison |
| [`docs/RUNNING.md`](docs/RUNNING.md) | Local run guide |

---

## Tech stack

**Python 3.11** · pandas · scikit-learn · **XGBoost** · TensorFlow/Keras · SHAP ·
**FastAPI** · Uvicorn · **Hopsworks** (feature store + model registry) ·
**GitHub Actions** · **React (Vite)** · Recharts · react-leaflet · Docker ·
Railway · Vercel · **Model Context Protocol**

Data: [Open-Meteo](https://open-meteo.com) — free, keyless, no registration.

---

<p align="center">
  Built for the <b>10Pearls Data Science Internship</b> · 22 cities · MIT licensed
</p>
