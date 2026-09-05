# Pearls AQI Predictor — Thesis Context & Source of Truth

> This document is the **authoritative reference** for writing the project thesis.
> It captures the *actual, current* system (which evolved well beyond the original
> README) and maps every fact to the thesis chapters. The thesis structure mirrors
> the sample in `Thesis_LaTeX/` (10 chapters), **minus** the university/front-matter
> details (no group members, supervisor, certificate, dedication, etc.) — only the
> content required to document the project.

---

## 0. One-line identity

**Pearls AQI Predictor** — an end-to-end, (near) 100% serverless machine-learning
system that forecasts the **Air Quality Index (AQI) up to 3 days (72 hours) ahead
for 22 cities across Pakistan**, retrains and self-monitors automatically, explains
its predictions, and serves them through a live React dashboard backed by a FastAPI
service.

Built for the **10Pearls Data Science Internship** (competitive; ~300–400
candidates, top-3 hired). The design goal was to go beyond "a model in a notebook"
to a **self-operating ML product** with real MLOps discipline.

**Live:**
- Frontend (Vercel): https://web-kappa-one-69x94ncqyn.vercel.app
- Backend API (Railway): https://aqi-backend-production-5af4.up.railway.app

---

## 1. IMPORTANT — scope evolution (read first)

The original brief/README described a **single-city (Lahore), Streamlit** demo. The
delivered system is substantially larger; the thesis must describe the **final
state**:

| Aspect | Original README | **Delivered (use this)** |
|---|---|---|
| Coverage | Lahore only | **22 cities**, all provinces + ICT + GB + AJK |
| Frontend | Streamlit | **React (Vite) + Recharts + react-leaflet**, custom "Aurora" design |
| Backend | (implicit in Streamlit) | **FastAPI** service |
| Model registry | mentioned | **Implemented** (Hopsworks Model Registry) |
| MLOps extras | — | Champion–Challenger gate, per-horizon eval, walk-forward backtest, SHAP→NL, What-If simulator, drift + forecast-error monitoring, MCP/LLM advisor |
| Deployment | Streamlit Cloud (planned) | **Vercel (frontend) + Railway (backend, Dockerfile)** — live |

---

## 2. Problem & motivation (→ Ch.1 Introduction, Ch.3 Problem)

- **The problem:** Pakistani cities (Lahore especially) are among the world's most
  polluted. Residents — particularly vulnerable groups (children, elderly,
  respiratory patients) — lack an accessible, forward-looking, *localized* air-quality
  forecast. Official AQI reporting is typically *nowcast* (current only), single-site,
  and not predictive.
- **Why forecasting (not nowcasting):** Actionable decisions (outdoor activity,
  school closures, mask use, medication) need *lead time*. A 3-day horizon lets people
  plan.
- **Why an ML *system* (not a one-off model):** Air quality is non-stationary
  (seasonal smog, weather-driven). A static model decays. The value is in a
  **continuously-running, self-monitoring, auto-retraining pipeline** — i.e. MLOps.
- **Motivation themes** (mirror the sample's three-part motivation style):
  1. *Public-health impact* — early warning for millions across 22 cities.
  2. *Engineering discipline* — demonstrate the full FTI/MLOps lifecycle on free,
     serverless infrastructure (reproducible, near-zero cost).
  3. *Trust & explainability* — every forecast is explained (SHAP → plain language),
     the model's promotion is gated and audited, and the system watches itself for drift.

**Problem statement (draft):** *Existing air-quality information for Pakistan is
current-only, single-site, unexplained, and manually maintained. There is a need for
an automated system that forecasts AQI several days ahead for many cities, retrains
and validates itself without human intervention, quantifies its own uncertainty and
drift, explains each prediction, and is served reliably to the public at negligible
cost.*

---

## 3. Aims & objectives (→ Ch.1)

**Aim:** Design, implement, deploy and evaluate a serverless, self-monitoring ML
system that forecasts 3-day AQI for 22 Pakistani cities and explains its predictions.

**Objectives:**
1. **Data & features** — automatically ingest weather + pollutant data (Open-Meteo)
   and engineer a robust, leakage-free feature set for many cities.
2. **Serverless FTI pipelines** — decoupled Feature / Training / Inference pipelines
   orchestrated by GitHub Actions + a Hopsworks Feature Store & Model Registry.
3. **Accurate multi-horizon forecasting** — a single global model predicting 1–72 h
   ahead, beating a persistence baseline at every horizon, with calibrated prediction
   intervals.
4. **MLOps discipline** — Champion–Challenger promotion gate, durable versioned model
   registry, per-horizon evaluation and walk-forward backtesting.
5. **Trust layer** — SHAP-based natural-language explanations, a What-If simulator,
   and self-monitoring (data drift via PSI + forecast-error tracking).
6. **Delivery** — a polished React dashboard + FastAPI backend + an LLM advisor,
   deployed live on free hosting.

---

## 4. Scope (→ Ch.1)

**In scope:** 22 cities; hourly feature pipeline; daily training; 72-h forecasts
(hourly + daily views); global XGBoost model; prediction intervals; feature store +
model registry; champion–challenger; per-horizon + walk-forward evaluation; SHAP
explanations; What-If simulator; drift + error monitoring; React dashboard; FastAPI
API; MCP/LLM advisor; live deployment.

**Out of scope:** training a bespoke foundation model; guaranteeing accuracy during
extreme unprecedented events; sub-hourly/street-level resolution; a paid data source;
mobile app. The system uses hosted models/APIs and free tiers.

---

## 5. Methodology / SDLC (→ Ch.1)

**Iterative & incremental**, module-by-module (matches how it was actually built):
Module 0 setup → Module 1 feature pipeline → Module 2 backfill → Module 3 EDA →
Module 4 training → Module 5 deep-learning exploration (LSTM) → Module 6 SHAP →
Module 7 dashboard (React+FastAPI) → Module 8 alerts → CI/CD → differentiators →
deployment. Each increment was independently testable; empirical results (e.g. a hung
materialization job, a datetime-dtype bug) fed the next iteration.

---

## 6. Architecture (→ Ch.5 Design)

**FTI architecture** (Feature–Training–Inference), the modern decoupled ML pattern,
glued by a **feature store** and **model registry**:

```
Open-Meteo API ──▶ Feature Pipeline ──▶ Hopsworks Feature Store ──▶ Training Pipeline ──▶ Model Registry
 (weather +         (hourly, GitHub                                  (daily, GitHub          │
  pollutants)        Actions)                                         Actions)               ▼
                          ▲                                                          Inference Pipeline
                          │                                                          (batch: predictions.json)
                    Backfill (one-off,                                                        │
                    2023→now history)                                                         ▼
                                                                            FastAPI backend ──▶ React dashboard
                                                                            (+ MCP/LLM advisor)
```

**Key architectural properties:**
- **Decoupling:** each pipeline runs independently on its own schedule; they
  communicate only through the feature store and model registry (no direct coupling).
- **Serverless & free:** compute = GitHub Actions; storage = Hopsworks free tier;
  hosting = Vercel + Railway. Near-zero running cost.
- **Stateless compute, durable state:** runners are ephemeral; all durable state
  (features, model, leaderboard) lives in the feature store / model registry / repo.
- **Multi-horizon *direct* forecasting:** the forecast horizon is a **feature**, so a
  single model serves all horizons (1–72 h) rather than 72 separate models.

---

## 7. Technology stack — justified (→ Ch.4/Ch.5)

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.11** | ML ecosystem; pinned `<3.12` for dependency compatibility |
| Data source | **Open-Meteo** (Air-Quality + Archive/Forecast Weather) | Free, **keyless**, historical (from ~2022-08) + forecast; global coverage |
| Feature store & model registry | **Hopsworks Serverless 5.0** (free tier) | Managed feature store + registry; industry-standard FTI backbone |
| Orchestration | **GitHub Actions** | Free serverless cron; already where the code lives |
| Classical ML | **scikit-learn, XGBoost** | XGBoost = champion; sklearn for baselines (Ridge, RandomForest, persistence) |
| Deep learning (explored) | **TensorFlow / Keras (LSTM)** | Evaluated as a sequence model; not promoted (XGBoost won) |
| Explainability | **SHAP** (TreeExplainer) | Per-prediction attributions → natural language |
| Backend | **FastAPI + Uvicorn** | Async, typed, fast; serves forecasts + on-demand compute |
| Frontend | **React (Vite), Recharts, react-leaflet** | Rich, interactive multi-page dashboard |
| LLM advisor | **Anthropic Claude + Model Context Protocol (MCP)** | Grounded air-quality Q&A over the system's own tools |
| Deployment | **Vercel** (frontend) + **Railway** (backend, Docker) | Free/cheap, CLI-deployable |

---

## 8. Data & features (→ Ch.5 Design §Data Design, Ch.6 Implementation)

### 8.1 Sources
- **Air-quality** (Open-Meteo Air-Quality API): `pm2_5, pm10, carbon_monoxide,
  nitrogen_dioxide, sulphur_dioxide, ozone, us_aqi` (hourly).
- **Weather** (Open-Meteo Archive/Forecast API): `temperature_2m,
  relative_humidity_2m, dew_point_2m, apparent_temperature, precipitation,
  surface_pressure, cloud_cover, wind_speed_10m, wind_direction_10m, wind_gusts_10m`.
- Merged per city/hour → **~21 raw columns**.

### 8.2 Target
- **AQI = Open-Meteo `us_aqi`** (US EPA AQI). Categories used across the UI: Good,
  Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous
  (with the standard colour scale and health advice).

### 8.3 Feature engineering (`src/aqi/features/engineering.py`)
- **Lag features** (previous hours), **rolling statistics** (windows), **calendar/
  cyclical features** (hour-of-day, day-of-week, month → sin/cos), **weather
  interactions**, **per-city anchor state** ("AQI now").
- Produces a **74-column** feature table per city.
- **Multi-horizon supervised set:** the horizon `h ∈ {1,2,3,6,12,24,36,48,60,72}` is
  added as a feature; targets are AQI shifted by `h`. Model trains on **27 model
  features** (subset used by the estimator).
- **Leakage control:** features at time *t* only use information available at *t*;
  chronological (never random) train/validation split.

### 8.4 Scale
- Full history **2023-01-01 → 2026-08** backfilled for all 22 cities:
  **~31,800 feature rows per city**, **~288,600 rows** read for training
  (subsampled to 200,000 for the model).

---

## 9. The pipelines (→ Ch.6 Implementation)

### 9.1 Feature pipeline — `src/aqi/pipelines/feature_pipeline.py`
- **Hourly** GitHub Action (`cron: 5 * * * *`). Fetches recent data
  (`past_days=7`) for all 22 cities, engineers features, **upserts** to Hopsworks
  (primary key `[city, timestamp]`, event time `datetime`).

### 9.2 Backfill — `src/aqi/pipelines/backfill.py`
- One-off `workflow_dispatch`. Loads full history in 3-month chunks per city.
- **Hardened after real failures:** non-blocking inserts (so a stuck free-tier
  Spark materialization can't freeze the run), and **history-aware resume** — skips a
  city only if its data already reaches back to `--start` (so recent-only rows from
  the hourly pipeline don't mask a city that was never fully backfilled).

### 9.3 Training pipeline — `src/aqi/pipelines/training_pipeline.py`
- **Daily** GitHub Action (`cron: 30 2 * * *`). Reads features, builds the
  multi-horizon supervised set, **chronological** split, trains candidates, selects
  the best (challenger), runs the **Champion–Challenger gate**, and — only if promoted
  — saves the model to the **Model Registry**. Also refreshes evaluation + monitoring
  snapshots and commits them.

### 9.4 Inference pipeline — `src/aqi/pipelines/inference.py`
- Loads the **champion from the Model Registry**, fetches each city's latest
  conditions + weather forecast, predicts the 72-h AQI curve with prediction
  intervals, attaches a **SHAP explanation** for the peak hour, logs forecasts (for
  later error scoring), and writes `predictions.json` served by the API/dashboard.

---

## 10. The model (→ Ch.6)

- **Global multi-horizon XGBoost regressor** (one model, all cities, all horizons).
- **Candidates compared each run:** Persistence baseline, Ridge, RandomForest,
  XGBoost. XGBoost wins.
- **Champion metrics (full-history, validation):** **RMSE 19.69, MAE 12.80,
  R² 0.85** (n_train ≈ 160k, n_valid ≈ 40k).
  - Baselines this run: Persistence 25.27 · Ridge 22.85 · RandomForest 20.38 →
    XGBoost 19.69 (best).
  - Earlier local-data champion: RMSE 20.597 — the full-history model *beat it* and
    was promoted (v1→v2), demonstrating the gate.
- **Per-horizon skill:** honest degradation with lead time — **+1 h R² ≈ 0.985 →
  +72 h R² ≈ 0.695**. NOTE: the model beats persistence at every horizon **from +2 h
  onward** (by 16–41%); at **+1 h persistence is better** (4.54 vs 6.10 RMSE). Report
  this honestly — the dashboard copy currently overstates it.
- **Walk-forward backtest:** 5 rolling time folds (TimeSeriesSplit), mean RMSE
  **20.55 ± 3.53** — the correct forward-performance estimate (always train on past,
  test on future).
- **Prediction intervals:** empirical residual quantiles per horizon (10th/90th →
  **80% interval**, split-conformal style), shown as the shaded band in the UI.

---

## 11. MLOps layer / the five differentiators (→ Ch.6, headline of the thesis)

1. **Champion–Challenger + Model Leaderboard** (`models/leaderboard.py`,
   `models/registry.py`, `models/hopsworks_registry.py`) — every run logs candidates
   to a persistent leaderboard; a challenger is **promoted only if it beats the
   champion's validation RMSE**; the champion model is stored durably in the
   **Hopsworks Model Registry** (`aqi_global_forecaster`, champion = lowest-RMSE
   version). Correctly **rejects a tie** (no improvement → keep incumbent).
2. **Per-horizon metrics + walk-forward backtesting** (`models/evaluate.py`) — RMSE/
   MAE/R² per lead time + 5-fold rolling backtest; figures + `evaluation.json`.
3. **SHAP → natural-language explanations** (`models/explain.py`) — per-instance
   SHAP phrased in plain English (e.g. *"Forecast AQI ≈ 200 (baseline 132); main
   drivers: current AQI +26, time of day +13.4, season −6.6"*). Surfaced in the
   dashboard and as an MCP tool.
4. **What-If Simulator** (`whatif.py`) — sliders for current PM2.5, AQI-now, wind,
   humidity, temperature → the model **re-predicts** and shows baseline vs scenario.
   (Explicitly labelled a *model simulation*, not causal proof.)
5. **Self-monitoring** (`monitoring.py`) — **data drift** via **Population Stability
   Index (PSI)** per feature (recent vs training; currently flags *significant*
   seasonal drift, temperature PSI ≈ 3.94), and **forecast-error tracking** (log each
   forecast, join to realized AQI, score MAE/RMSE, auto-flag biggest misses).

**Plus:** an **MCP / LLM Air-Quality Advisor** — an Anthropic-Claude chatbot grounded
in the system's own tools (forecast lookup, explanation, etc.) via the Model Context
Protocol; also exposed as a chat panel in the dashboard. NOTE: `/api/chat` currently
returns 503 in production because `ANTHROPIC_API_KEY` is not set on Railway.

---

## 12. Frontend / UI (→ Ch.7 UI)

- **React (Vite)** SPA with a custom **"Aurora"** dark design system (glassmorphism,
  cyan→teal gradient accent chosen to complement the black-and-white 10Pearls logo,
  animated aurora background, 3D hover, animated intro/splash, round logo badge).
- **Four tabs:** **Forecast** (city selector, current-AQI card, 72-h/daily chart with
  prediction-interval band, Pakistan **map** (react-leaflet) coloured by AQI, legend,
  alert banner, SHAP explanation card, chat advisor), **Model Evaluation**
  (leaderboard table + per-horizon line chart + walk-forward table), **Monitoring**
  (PSI drift table + forecast-error/biggest-misses), **What-If** (driver sliders →
  baseline-vs-scenario).
- Charts: **Recharts**; responsive; theme-consistent.
- **Static-mode fallback:** when no backend URL is configured, the app reads
  committed JSON snapshots (`/data/*.json`) so it can run as a pure static site
  (What-If + Chat hidden).

---

## 13. Deployment (→ Ch.6 §Deployment / Ch.8)

- **Frontend → Vercel** (Vite build; `VITE_API_URL` baked into the build command
  since Vercel blocks `VITE_`-prefixed env vars as "sensitive").
- **Backend → Railway** via a **Dockerfile** (Railway now defaults to *Railpack*,
  which ignored `nixpacks.toml`/`railway.json`; a root Dockerfile is auto-detected and
  gives full control). Slim image (`requirements-deploy.txt`: no hopsworks/tensorflow),
  editable install so `PROJECT_ROOT` resolves to `/app` where the committed data +
  model bundle live.
- **The deployment journey (good "challenges" material):** Railway trial expired on
  the first account → 2nd account; Railpack vs Nixpacks builder; `railway up` honours
  `.gitignore` and dropped the committed model/data (1.1 MB snapshot →
  `predictions_ready:false`) until `.gitignore` negations force-shipped them; Vercel
  `VITE_` env-var restriction.
- **Two deployment modes** exist: (a) **static** (Vercel-only, backend-less, reads
  committed JSON) and (b) **dynamic** (Vercel + live Railway FastAPI, all features
  including live What-If/SHAP; Chat needs an `ANTHROPIC_API_KEY`). The system is
  currently deployed in **dynamic** mode.

---

## 14. Results & evaluation (→ Ch.8 Testing/Evaluation)

- **Accuracy:** champion XGBoost RMSE 19.69 / MAE 12.80 / R² 0.850; beats persistence,
  Ridge, RandomForest.
- **Per-horizon:** R² 0.985 (+1 h) → 0.695 (+72 h); beats the baseline at every
  horizon from +2 h onward (not at +1 h — see §10).
- **Walk-forward:** mean RMSE 20.55 (± 3.53) over 5 folds.
- **LSTM study:** LSTM 22.12 vs XGBoost 22.63 vs persistence 28.79 at the +24 h task —
  the LSTM edges it by 2.3% at that single horizon but was not promoted (coverage,
  25 min vs 1 min training cost, explainability).
- **Calibration:** 80% prediction intervals from residual quantiles.
- **Drift:** PSI flags significant seasonal drift (a *correct retrain signal*).
- **System validation:** 13 endpoints exercised against the live deployment — 12
  return 200 in 0.72–1.51 s; `/api/chat` returns its documented 503.
- **Unit tests** (`tests/`): **12 passed** — covering the EPA AQI computation and the
  feature / supervised builders (including the two leakage guards).

---

## 15. Challenges & solutions (→ Ch.6/Ch.8, great thesis material)

| Challenge | Root cause | Solution |
|---|---|---|
| Hopsworks CI failures (project-not-found, DELTA lib, Kafka, stats OOM) | Serverless 5.0 quirks + free-tier limits | Correct project name; `time_travel_format="HUDI"`; `hopsworks[python]` extra; `statistics_config=False` + fresh feature-group v2 |
| Windows can't install `hopsworks` | `twofish` build failure | Run all Hopsworks work on Linux CI; local dev uses Parquet fallback backend |
| Backfill froze at city 10/22 | Blocking insert waited forever on a stuck free-tier Spark job | Non-blocking inserts + history-aware resume (idempotent upserts) |
| Backfill "skipped" already-loaded cities that weren't fully loaded | Resume checked *presence*, but hourly pipeline had seeded recent rows | Resume now checks *earliest date reaches `--start`* |
| Training crashed on `time_split` | Hopsworks returns `datetime` as strings | Coerce to datetime on read + in the split |
| Champion/leaderboard reset every CI run | Stored in ephemeral `models_local/` | Persist model to Hopsworks Model Registry + commit `leaderboard.json` |
| Railway build failed | Railpack default ignored nixpacks/railway.json; no start command | Dockerfile (auto-detected) |
| Backend served no data | `railway up` honoured `.gitignore`, dropped committed artifacts | `.gitignore` negations to force-ship model + data JSON |
| XGBoost failed to import in the container | Slim base image lacks the OpenMP runtime | `apt-get install libgomp1` in the Dockerfile |
| App resolved data paths to the wrong directory in the container | Ordinary install put the package in site-packages, so `PROJECT_ROOT` pointed away | Editable install (`pip install -e . --no-deps`) inside the image |

---

## 16. Mapping: project content → thesis chapters

| Chapter (from `Thesis_LaTeX/`) | Content to fill from this doc |
|---|---|
| **Ch.1 Introduction** | §0,2,3,4,5,6 (brief, motivation, aims, scope, SDLC, architecture-at-a-glance) |
| **Ch.2 Literature Review** | Air-quality forecasting methods (statistical vs ML vs DL); AQI standards; feature stores / FTI architecture; MLOps (champion–challenger, drift/PSI); SHAP/explainable AI; time-series forecasting & walk-forward validation. *(needs citations)* |
| **Ch.3 Problem Definition** | §2 (expanded), gaps in existing AQI tools (nowcast-only, single-site, unexplained, manual) |
| **Ch.4 Requirements** | Functional (ingest, store, train, evaluate, forecast, explain, monitor, serve, advise) + non-functional (serverless, reproducible, low-cost, resilient, explainable); use cases; §7 tech choices |
| **Ch.5 Design** | §6 architecture, §7 stack, §8 data design, FTI diagram, component/sequence/deployment diagrams |
| **Ch.6 Implementation** | §9 pipelines, §10 model, §11 differentiators, §13 deployment, §15 challenges |
| **Ch.7 User Interface** | §12 (tabs, components, Aurora design, screenshots) |
| **Ch.8 Testing & Evaluation** | §14 results, unit tests, endpoint validation, per-horizon + walk-forward + drift |
| **Ch.9 Conclusion & Future Work** | Contributions; limitations; future (real-time street-level, more DL, causal what-if, mobile, more countries) |
| **Ch.10 Glossary & References** | §17 abbreviations + glossary + bibliography |

---

## 17. Figures produced (→ `Thesis/figures/`, HTML→PNG like the sample)

1. `architecture` — FTI serverless architecture (data → feature store → training →
   registry → inference → dashboard).
2. `fti-pipelines` — the three decoupled pipelines + schedules.
3. `feature-engineering` — raw → lag/rolling/cyclical → multi-horizon supervised set.
4. `multihorizon` — horizon-as-feature, single global model.
5. `champion-challenger` — promotion-gate flow + leaderboard + registry.
6. `error-by-horizon` — per-horizon RMSE/R² (generated by `evaluate.py`).
7. `walk-forward` — rolling backtest folds (generated).
8. `monitoring` — PSI drift + realised-error loop.
9. `shap-importance` — global feature attribution.
10. `deployment` — Vercel + Railway topology, static vs dynamic modes.
11. `ui-forecast`, `ui-eval`, `ui-monitoring`, `ui-whatif` — real screenshots of the
    live dashboard.
12. `sdlc` — iterative module-by-module lifecycle.
13. `sequence-inference` — request → registry load → forecast → SHAP → JSON → UI.
14. `usecase`, `component`, `data-design` — UML-style use case, module decomposition,
    and data design.
15. `eda-city-ranking`, `eda-seasonality`, `eda-diurnal`, `eda-weather-correlation`,
    `eda-aqi-distribution` — exploratory analysis figures.

---

## 18. Abbreviations & glossary (→ Ch.10)

AQI (Air Quality Index) · PSI (Population Stability Index) · FTI (Feature–Training–
Inference) · MLOps (Machine-Learning Operations) · SHAP (SHapley Additive
exPlanations) · MCP (Model Context Protocol) · RMSE/MAE/R² (error metrics) ·
PM2.5/PM10 (particulate matter) · CO/NO₂/SO₂/O₃ (pollutant gases) · HUDI (Hopsworks
time-travel table format) · CI/CD · API · SPA · LSTM (explored) ·
XGBoost (Extreme Gradient Boosting) · US EPA AQI (the AQI standard used).

---

## 19. Quick-reference numbers (keep consistent across the thesis)

- **22** cities · all 4 provinces + **ICT + GB + AJK**
- Horizon: **72 hours (3 days)**; horizons modelled: **1,2,3,6,12,24,36,48,60,72 h**
- History: **2023-01-01 → 2026-08**; **696,960** total hourly rows;
  **200,000** subsampled for training; **160,002 train / 39,998 valid**
- Raw features **~21 cols → 74** engineered; **27** model features
- Champion **XGBoost**: **RMSE 19.69 · MAE 12.80 · R² 0.850**
- Baselines: Persistence 25.27 · Ridge 22.85 · RandomForest 20.38
- Per-horizon R²: **0.985 (+1h) → 0.695 (+72h)**; walk-forward mean RMSE **20.55**
- EDA: most polluted **Faisalabad 156.9**, then Lahore 151.5; cleanest **Gilgit 75.7**;
  worst month **January**; strongest weather driver **surface_pressure (r = 0.32)**
- Prediction interval: **80%** (10th/90th residual quantiles)
- Schedules: feature **hourly** (`5 * * * *`), training **daily** (`30 2 * * *`)
- Registry model: **`aqi_global_forecaster`**; feature group **`aqi_features` v2**
- Live: frontend `web-kappa-one-69x94ncqyn.vercel.app`, backend
  `aqi-backend-production-5af4.up.railway.app`

---

## 20. Writing conventions for the thesis

- Follow the sample's **academic, formal, third-person** tone; each chapter opens with
  a brief and closes with a summary.
- Use the template's macros: `\fig{file}{width}{caption}{label}`, `\projname`
  (set to "Pearls AQI Predictor"), `xltabular` for wide tables, `\cite{}` for refs.
- **Skip** all university front-matter (supervisor, certificate, declaration,
  dedication) — the title page carries only the 10Pearls logo, the project title and
  **Submitted by: Hasnain Irshad**.
- The delivered thesis lives in **`Thesis/`** (127 pages, 35 figures, 23 tables,
  44 references). `Thesis_LaTeX/` was the reference sample.
