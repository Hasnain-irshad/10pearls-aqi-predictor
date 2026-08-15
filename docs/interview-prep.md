# 🎤 Pearls AQI Predictor — Interview Preparation & Project Documentation

> **Purpose.** This is the master study document for defending the AQI Predictor
> project in interviews. It captures **every module** — the *what*, the *why*,
> the trade-offs, the code decisions, and likely interview questions.
> It is updated after each module so nothing is left out.
>
> **How to build the PDF** (at the end): see [Appendix A](#appendix-a--building-the-pdf).

**Project:** Pearls AQI Predictor · **Intern:** Hasnain Irshad · **Org:** 10Pearls
**City forecast:** Lahore, Pakistan · **Horizon:** next 3 days

---

## 📑 Table of Contents
1. [30-Second Elevator Pitch](#1-30-second-elevator-pitch)
2. [The Problem & Why It Matters](#2-the-problem--why-it-matters)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack — Every Choice Justified](#4-tech-stack--every-choice-justified)
5. [Core Concepts Glossary](#5-core-concepts-glossary)
6. [Module 0 — Foundation & Project Setup](#6-module-0--foundation--project-setup)
7. [Module 1 — Feature Pipeline](#7-module-1--feature-pipeline)
8. [Module 2 — Historical Backfill](#8-module-2--historical-backfill) *(pending)*
9. [Module 3 — Exploratory Data Analysis](#9-module-3--eda) *(pending)*
10. [Module 4 — Training Pipeline](#10-module-4--training-pipeline) *(pending)*
11. [Module 5 — Deep Learning Model](#11-module-5--deep-learning) *(pending)*
12. [Module 6 — Explainability (SHAP)](#12-module-6--shap) *(pending)*
13. [Module 7 — Dashboard](#13-module-7--dashboard) *(pending)*
14. [Module 8 — Alerts](#14-module-8--alerts) *(pending)*
15. [CI/CD Automation](#15-cicd-automation) *(pending)*
16. [General / Behavioral Interview Questions](#16-general--behavioral-questions)
17. [Appendix A — Building the PDF](#appendix-a--building-the-pdf)

---

## 1. 30-Second Elevator Pitch

> "I built an end-to-end, fully serverless machine-learning system that forecasts
> Lahore's Air Quality Index three days ahead. It automatically collects weather
> and pollution data every hour, engineers features into a feature store, retrains
> models daily, and serves live forecasts through a public dashboard — all running
> on free infrastructure (GitHub Actions + Hopsworks + Streamlit) with no server to
> manage. I compared statistical, tree-based, and deep-learning models, explained
> predictions with SHAP, and added health alerts for hazardous days."

**Why this pitch works:** it names the impact (health), the scope (end-to-end),
the engineering maturity (serverless, automated, explainable), and the rigor
(multiple model families).

---

## 2. The Problem & Why It Matters

- Lahore is repeatedly ranked among the **most polluted cities on Earth**; AQI
  regularly exceeds 150–200 (Unhealthy) and spikes far higher in winter smog.
- A 3-day forecast lets residents — especially children, elderly, and people with
  respiratory conditions — **plan ahead** (mask up, limit outdoor exertion).
- **ML framing:** this is a **time-series regression / forecasting** problem —
  predict a continuous AQI value at future time steps from weather + pollution
  history and calendar features.

---

## 3. System Architecture

```
 ┌──────────────┐   hourly    ┌──────────────────┐        ┌─────────────────────┐
 │ Open-Meteo   │────────────▶│ FEATURE PIPELINE │───────▶│  Hopsworks          │
 │ weather +    │  raw data   │ fetch→compute→   │features│  Feature Store      │
 │ pollutants   │             │ store            │        │  (single source of  │
 └──────────────┘             └──────────────────┘        │   truth)            │
                                                          │        │            │
                              ┌──────────────────┐  daily │        ▼            │
                              │ TRAINING PIPELINE │◀───────┤  features + targets │
                              │ train→evaluate→   │        │                     │
                              │ register best     │───────▶│  Model Registry     │
                              └──────────────────┘  model  └─────────┬───────────┘
                                                                     │ model + features
                                                                     ▼
                                                          ┌─────────────────────┐
                                                          │  Streamlit Dashboard │
                                                          │  live + 3-day AQI,   │
                                                          │  SHAP, alerts        │
                                                          └─────────────────────┘
```

**The four pipelines (memorize this):**
1. **Feature pipeline** (hourly) — turns raw API data into stored features.
2. **Backfill** (one-off) — populates history so we have training data.
3. **Training pipeline** (daily) — trains, evaluates, registers the best model.
4. **Inference/app** (on demand) — loads model + features, shows forecasts.

**Why decouple them?** Each can run, fail, and scale independently. The feature
store is the clean interface between them — no pipeline needs to know how another
works. This is the "**FTI (Feature/Training/Inference) architecture**."

---

## 4. Tech Stack — Every Choice Justified

| Component | Choice | Why this one | Alternatives & trade-off |
|-----------|--------|--------------|--------------------------|
| **Data API** | Open-Meteo | Free, **no API key**, gives history **+** forecast for weather & air quality | AQICN/OpenWeather need keys (secret management); some cap history |
| **Feature Store** | Hopsworks | Free tier, purpose-built for serverless ML, includes Model Registry | Vertex AI (needs GCP billing, heavier); plain CSV (no versioning/upsert) |
| **Orchestration** | GitHub Actions | Truly serverless, free, version-controlled, visible run history | Airflow (needs a server to host — not serverless); cron on a VM (not free) |
| **Classical ML** | scikit-learn + XGBoost | Industry standard; XGBoost strong on tabular | LightGBM (similar); pure statsmodels (weaker on many features) |
| **Deep Learning** | TensorFlow (LSTM) | Handles temporal sequences; required by brief | PyTorch (equally valid); Prophet (statistical, less flexible) |
| **Explainability** | SHAP | Game-theoretic, model-agnostic, rich plots | LIME (local only, less consistent) |
| **Dashboard** | Streamlit | Pure-Python, fast to build, free public hosting | Gradio (ML-demo focused); Flask/Dash (more code) |
| **Language/Env** | Python 3.11 + conda | 3.11 = best compatibility for TF + Hopsworks | 3.13 too new (TF/Hopsworks lag) |

> **Interview line:** *"Every tool was chosen so the whole system stays free and
> serverless — the brief asked for a 100% serverless stack, and I can point to
> exactly where each requirement is satisfied."*

---

## 5. Core Concepts Glossary

Concepts an interviewer will probe. Be able to explain each in one or two sentences.

| Concept | Explanation |
|---------|-------------|
| **AQI (Air Quality Index)** | A 0–500 index *computed* from pollutant concentrations. Each pollutant → a sub-index via a piecewise-linear formula; overall AQI = **max** of sub-indices (worst pollutant wins, because health risk is driven by the worst one). |
| **Feature** | A model input derived from raw data (e.g. "24-hour rolling mean PM2.5"). |
| **Target / label** | What we predict (future AQI). |
| **Feature Store** | A versioned database of features that is the single source of truth for both training and serving — prevents **train/serve skew**. |
| **Train/serve skew** | When features are computed differently in training vs production, silently degrading the live model. A feature store eliminates it. |
| **Target leakage** | When a feature accidentally contains information from the future/target, making test scores look great but production fail. Fixed by `.shift()`-ing lag/rolling features so they only use past rows. |
| **Cyclical encoding** | Encoding periodic features (hour, month) as `(sin, cos)` so the model knows hour 23 ≈ hour 0 (they're adjacent on a circle). |
| **Lag feature** | A past value of a series (AQI 24h ago) used to predict the present/future — leverages autocorrelation. |
| **Rolling feature** | A statistic over a moving window (24h mean/std), capturing recent trend. |
| **Walk-forward validation** | Time-series-correct evaluation: always train on the past, test on the future. Never a random split (that leaks the future). |
| **Idempotency** | Re-running the pipeline produces the same result (no duplicate rows) — achieved via **upsert** on a primary key. |
| **Model Registry** | Versioned store of trained models + their metrics, so the app always loads the current best. |
| **Baseline model** | A trivial model (e.g. "tomorrow = today", persistence) that real models must beat to justify their complexity. |

---

## 6. Module 0 — Foundation & Project Setup

**Goal:** a professional, reproducible repository.

**What was built:**
- `src/` layout with an installable `aqi` package (`pip install -e .`) → clean
  imports (`from aqi.config import LOCATION`) in local, CI, and the app.
- **Split requirements** (`requirements.txt` core / `-app` / `-dl` / `-dev`) so
  the hourly CI job doesn't waste minutes installing TensorFlow it doesn't need.
- Secrets via `.env` (gitignored) locally and **GitHub Secrets** in CI — never
  committed.
- `.gitignore`, `.gitattributes` (LF/CRLF normalization), MIT `LICENSE`,
  `pyproject.toml`, architecture-rich `README.md`.
- Central `config.py` — one source of truth for city coordinates, Hopsworks
  names, and forecast horizon.

**Why it matters for scoring:** reviewers skim. A clean, engineered structure
signals "software engineer," not "notebook hacker," before they read any ML code.

**Interview Q&A:**
- *Why a `src/` layout + editable install?* → Guarantees the same import path
  everywhere; avoids fragile `sys.path` hacks; makes the code a real package.
- *Why split requirements?* → Faster, cheaper CI; the app image stays small.
- *How do you handle secrets?* → `.env` locally (gitignored), GitHub Secrets in
  CI, injected as environment variables at runtime. Nothing sensitive in git.

---

## 7. Module 1 — Feature Pipeline

**Goal:** raw API data → engineered features → feature store, hourly & automated.

### 7.1 The three jobs
1. **Fetch** raw weather + pollutant data (Task 1.1).
2. **Compute** AQI target + features (Tasks 1.2–1.3).
3. **Store** in Hopsworks (Task 1.4).

### 7.2 Task 1.1 — Fetching (`src/aqi/data/openmeteo.py`) ✅

**What it does:** `fetch_air_quality()` calls Open-Meteo's air-quality endpoint
for Lahore and returns a tidy hourly DataFrame (PM2.5, PM10, CO, NO₂, SO₂, O₃,
plus Open-Meteo's own `us_aqi`).

**Five design decisions (be ready to defend each):**
1. **Coordinates from `config.py`, not hardcoded** — DRY; extending to another
   city is a one-line change.
2. **Retry-resilient HTTP session** (`Retry` + exponential backoff on
   429/5xx) — the job runs unattended hourly; a transient blip must not crash it.
   *This is the production-maturity signal most interns miss.*
3. **`raise_for_status()`** — fail fast and loud at the source of the error,
   not with a confusing `KeyError` later.
4. **Defensive parsing** — verify `hourly.time` exists before trusting the
   response; never assume an external API's shape.
5. **Explicit `timezone=Asia/Karachi`** — returns Lahore local time. Forgetting
   this silently returns UTC and shifts every hour-of-day feature by 5 hours —
   a subtle, model-wrecking bug.

**Verified live:** 72 rows fetched for Lahore; AQI ≈ 171–172 (Unhealthy) — a
realistic value that sanity-checks the whole fetch path.

**Interview Q&A:**
- *Why Open-Meteo over AQICN/OpenWeather?* → Free, keyless, history + forecast.
- *What if the API is down during the hourly run?* → Retries with backoff; if it
  still fails, the job errors and GitHub flags a red run (observability).
- *Why fetch 7 `past_days` when you only need the latest hour?* → Lag & rolling
  features for the newest rows need recent history to be computed correctly.

### 7.3 Task 1.2 — AQI Computation (`src/aqi/data/aqi.py`) ✅

**What it does:** computes the US EPA AQI from pollutant concentrations, provides
the 6 health categories, and a `is_hazardous()` check for alerts.

**The formula (know it cold):** for a reading `C` in concentration bin
`[C_lo, C_hi]` mapping to index bin `[I_lo, I_hi]`:
> AQI = (I_hi − I_lo) / (C_hi − C_lo) × (C − C_lo) + I_lo

Each pollutant → a sub-index; **overall AQI = MAX of sub-indices** (the worst
pollutant defines air quality, because health risk follows the worst offender).

**⭐ The bug story (tell this in the interview):** My first version computed AQI
across *all* pollutants and returned **500 (Hazardous)** when PM2.5 was only
~48 µg/m³ and Open-Meteo said ~85. Cross-checking against Open-Meteo's `us_aqi`
exposed it. **Root cause:** Open-Meteo reports **CO in µg/m³**, but my CO
breakpoint table was in **mg/m³** (max 57.5) — urban CO (~300 µg/m³) blew past
it and hit the "above top breakpoint → 500" cap. **A unit-mismatch bug.**
**Fix:** compute AQI from **PM2.5 + PM10 only** (units we trust); gas tables
kept for reference but excluded (`POLLUTANTS_FOR_AQI`), with proper gas unit
conversion listed as a documented future improvement.
> *Lesson: always cross-validate a computation against an independent source.*

**Second subtlety (not a bug):** even after the fix, our value (~130) sits above
Open-Meteo's (~85) for the same hour. Reason: EPA's PM2.5 AQI uses a **24-hour
average**; we use the *instantaneous hourly* value, which runs hotter. Because we
use Open-Meteo's properly-averaged `us_aqi` as the actual model **target**, our
`compute_aqi` is only for labels/education — so this is acceptable and documented.

**Interview Q&A:**
- *Why is overall AQI the max, not the mean?* → Health risk is driven by the
  single worst pollutant; averaging would hide a dangerous spike.
- *Why did your AQI read 500?* → Unit mismatch on CO (µg/m³ vs mg/m³); found via
  cross-check; fixed by restricting to trusted PM units.
- *Why not just use Open-Meteo's us_aqi and skip your own?* → We do use it as the
  target; computing our own documents what the number means, powers category
  labels/alerts, and lets us turn *predicted concentrations* into an AQI.

### 7.4 Task 1.3 — Feature Engineering (`src/aqi/features/engineering.py`) ✅

**What it does:** turns 19 raw columns into **72 features**. Groups:
- **Time:** hour, day, month, day-of-week, day-of-year, is_weekend.
- **Cyclical (sin/cos):** hour, month, day-of-week encoded on a circle so 23:00 ≈
  00:00. *Why:* a raw 0–23 hour tells the model 23 and 0 are 23 apart when they're
  adjacent; sin/cos fixes that.
- **Wind vectors (u/v):** decompose speed+direction into east-west / north-south
  components. *Why:* direction is circular (359° ≈ 1°); u/v makes wind linear-friendly.
- **Lag features:** AQI/PM2.5/PM10 at 1, 3, 6, 12, 24 h ago. *Why:* air quality is
  autocorrelated — the recent past predicts the near future.
- **Rolling stats:** 6h & 24h mean/std/max. *Why:* summarise recent trend & volatility.
- **AQI change rate:** `diff` and relative change — pollution momentum (brief-required).

**⭐ THE key concept — target leakage:** every rolling window is `.shift(1)`-ed so
it ends at the *previous* hour and never includes the current row. Without this,
a "feature" would contain the answer, giving fake-great test scores that collapse
in production. **This is the single most important detail in the whole pipeline.**

**Design choices:**
- **Target = Open-Meteo `us_aqi`** (properly averaged), falling back to our
  `compute_aqi` only where missing.
- **Primary key = epoch-seconds `timestamp`**, computed with a version-robust
  `(dt - epoch) // 1s` (plain `.astype(int64)` on datetimes is deprecated in
  pandas 3.x).
- NaNs in lag/rolling columns appear only at the very start of the series
  (e.g. 24 NaNs for the 24h lag) — expected and correct.

**Interview Q&A:**
- *What is target leakage and how did you prevent it?* → A feature containing
  current/future info; prevented by `.shift(1)` on all rolling/lag features.
- *Why sin/cos for hour?* → To represent periodicity; 23:00 and 00:00 become adjacent.
- *Why decompose wind into u/v?* → Direction is circular; components are linear and
  physically meaningful (which way pollutants are pushed).
- *Why lag features at all?* → AQI is autocorrelated; yesterday strongly predicts today.

### 7.5 Task 1.4 — Storing to Hopsworks *(pending)*
> Feature group, primary key `(city, timestamp)`, event time, upsert/idempotency,
> and the Windows `twofish` build issue + chosen workaround.

---

## 8. Module 2 — Historical Backfill

**What it does:** replays the feature logic over ~3.6 years (2023-01 → 2026-08)
for all 22 cities, producing **696,960 hourly feature rows** — the training set.

**Key design points:**
- **Chunked fetching:** history is pulled in ~3-month chunks (`_month_chunks`) to
  keep each API request small and reliable; chunks are concatenated *per city*.
- **Two weather endpoints:** recent data uses Open-Meteo's forecast endpoint;
  historical ranges use the **ERA5 archive endpoint** (`archive-api`). The air-
  quality endpoint serves both via `start_date`/`end_date`.
- **Features built on a contiguous per-city series** so lags/rolling windows are
  correct and never cross a city boundary.
- **Fault-tolerant:** if one city's fetch fails, the loop logs it and continues.

**The storage abstraction (important architecture point):** all pipelines call
`aqi.data.store.save_features()`, which writes to **local Parquet** by default and
to **Hopsworks automatically when `HOPSWORKS_API_KEY` is set**. This let the whole
system be built and validated locally, with zero code change needed to switch to
the Feature Store.

**Multi-city / global-model design:**
- A single `CITIES` registry (config) with province tags is the one source of truth.
- One **global model** trained on all cities uses `latitude`, `longitude`, and
  weather as features, so it generalises — and can predict for **any** location on
  demand, not just the 22 it trained on.

**Interview Q&A:**
- *Why backfill at all?* → A model needs history; the hourly pipeline only adds
  new rows going forward.
- *Why local Parquet AND Hopsworks?* → Decoupling via a storage interface means
  development isn't blocked on cloud credentials, and there's no train/serve skew.
- *One global model or one per city?* → Global: generalises, scales to new cities,
  fewer artifacts; city identity is captured by location + weather features.

## 9. Module 3 — EDA

**What it does:** generates 5 figures + a findings file (`docs/eda_findings.md`)
from the 697k-row dataset.

**Key findings (real data, 2023–2026):**
- **Most polluted city: Faisalabad** (mean AQI ≈ 157) — an industrial hub, it
  edges out Lahore on the long-run average.
- **Cleanest: Gilgit** (mean AQI ≈ 76) — mountain air.
- **Seasonality: January is the worst month** — the winter-smog spike is stark
  (temperature inversions trap pollutants). This justifies the `month` cyclical feature.
- **Diurnal pattern:** AQI varies by hour of day → justifies the `hour` cyclical feature.
- **Strongest weather correlate:** surface pressure (r ≈ 0.32); wind and
  temperature also matter → justifies the weather features.

**Figures:** `eda_aqi_distribution`, `eda_city_ranking`, `eda_seasonality`,
`eda_diurnal`, `eda_weather_correlation` (in `docs/images/`).

**Interview Q&A:**
- *What did EDA tell you that shaped the model?* → Strong monthly + hourly
  seasonality and weather dependence → cyclical time features + weather features.
- *Why is January worst?* → Winter temperature inversions + low wind trap
  pollutants near the surface.

## 10. Module 4 — Training Pipeline

**The forecasting design (know this cold — it's the cleverest part):**
We predict AQI at a future hour `τ = t + h` (h = 1…72). At time `t` we know:
- **Target-time features** — the *forecasted* weather at `τ`, the calendar at `τ`
  (deterministic), and the location. (Open-Meteo gives us the weather forecast.)
- **Anchor-state features** — the latest observed pollution at `t` (current AQI,
  recent rolling means, current PM), obtained by shifting each series by `h`.
- **`horizon` (h) itself is a feature** → **one** global model serves every city
  and every lead time from +1h to +72h.

**Why not just autoregress?** Recursively feeding predictions back in compounds
errors. Our direct, weather-driven approach avoids that and exploits the fact
that weather (which drives dispersion) is itself forecastable.

**Honest evaluation:**
- **Chronological split** (`time_split`) — train on the past, validate on the most
  recent 20%. A random split would leak the future and inflate scores.
- **Persistence baseline** — "AQI in h hours = AQI now." Real models must beat it,
  which justifies their complexity.
- **Models compared:** Ridge (linear), RandomForest, XGBoost (gradient boosting).
- **Metrics:** RMSE (penalises big misses), MAE (average error, same units as AQI),
  R² (variance explained). See `docs/model_metrics.md` for the results table.

**Prediction intervals:** we take the validation residuals *per horizon* and use
their 10th/90th percentiles as an 80% interval (a split-conformal-style method).
So the dashboard shows a *band*, not a false-precision single line — and the band
correctly widens at longer horizons.

**Documented assumption:** training uses the *actual* weather at `τ`; inference
uses the *forecast*. We assume the weather forecast is good (Open-Meteo's is) —
standard practice, and stated as a limitation.

**Interview Q&A:**
- *Why is `horizon` a feature?* → It lets one model cover all lead times and learn
  how uncertainty/behaviour changes with distance into the future.
- *Why a persistence baseline?* → To prove the ML actually adds value over the
  trivial "nothing changes" forecast.
- *How do you get uncertainty from a point model?* → Empirical residual quantiles
  per horizon (conformal-style intervals).
- *Why RMSE and MAE?* → RMSE punishes large errors (dangerous AQI spikes matter
  more); MAE is the interpretable average error.

## 11. Module 5 — Deep Learning (LSTM)

**What it does:** a TensorFlow **LSTM** consumes the past 48 hours of pollution +
weather (a real *sequence*, per city) and predicts AQI 24h ahead — the
deep-learning member of the model family the brief asks for. Global model,
chronological split, compared fairly against XGBoost on the *same* +24h task.

**Real results (+24h):**
| Model | RMSE | MAE | R² |
|-------|-----:|----:|---:|
| LSTM | **22.12** | 14.32 | 0.799 |
| XGBoost (same task) | 22.63 | 14.48 | 0.789 |
| Persistence baseline | 28.79 | 17.21 | 0.660 |

**The nuanced finding (great interview material):** the LSTM *slightly edges out*
XGBoost at the 24-hour horizon, but **XGBoost remains the production model** — it
covers all horizons 1–72h in one model (overall RMSE 20.6), trains in ~1 min vs
~25 min, and is SHAP-explainable. *We chose by measuring, not by hype.*

**Interview Q&A:**
- *Why did you build an LSTM if XGBoost ships?* → The brief asks for statistical →
  deep-learning variety; and the comparison is itself a finding — a sequence model
  is competitive but not worth its cost here.
- *Why does XGBoost win overall despite the LSTM edging it at +24h?* → One
  gradient-boosted model handles every lead time and is far cheaper to train/serve.

## 12. Module 6 — SHAP

**What it does:** explains the model with Shapley values (each feature's fair
contribution to each prediction). `TreeExplainer` gives exact, fast values for the
XGBoost model; results saved to `docs/images/shap_importance.png`.

**What drives the forecast (real results, ranked):**
1. `aqi_anchor` — the current AQI (by far the strongest; air quality is autocorrelated).
2. `pm2_5_anchor` — current PM2.5 (the dominant pollutant).
3. `aqi_roll_mean_24h` — recent 24-hour trend.
4. `hour_sin`, `month_cos` — time of day and season.
5. `horizon` — how far ahead we're predicting.
6. `latitude`/`longitude` — **the global model genuinely uses geography** to
   differentiate cities.
7. `surface_pressure` — the top weather driver.

**Interview Q&A:**
- *What are SHAP values?* → A game-theoretic attribution: each feature's average
  marginal contribution to a prediction across all feature orderings.
- *What did SHAP confirm?* → The model behaves sensibly — it leans on current
  pollution + recent trend + time/season + location + weather, not spurious signals.

## 13. Module 7 — Dashboard (React + FastAPI)

**Architecture:** a **FastAPI** backend serves pre-computed forecasts (and can run
on-demand predictions for any location); a **React** (Vite) frontend renders them.
Keeping inference a scheduled *batch* job (writing `predictions.json`) means the
core dashboard needs no always-on server — it stays serverless.

**Frontend features:** city dropdown grouped by province, hourly/daily toggle, a
forecast chart with the **prediction-interval band**, an interactive **Leaflet map**
of Pakistan (circles coloured by live AQI), a hazardous-air **alert banner**, and
the EPA colour legend.

**API endpoints:** `/api/health`, `/api/cities`, `/api/categories`,
`/api/predictions`, `/api/predictions/{city}`, `/api/predict` (on-demand, any lat/lon).

**Interview Q&A:**
- *Why React + FastAPI instead of Streamlit?* → A far more polished, interactive UI;
  FastAPI satisfies the brief's "Flask/FastAPI" option; and pre-computed
  predictions keep it serverless.
- *How does "any city" work with a fixed training set?* → The global model uses
  location + weather features, so `/api/predict?lat=&lon=` forecasts anywhere.

## 14. Module 8 — Alerts

**What it does:** `check_forecast()` scans a city's 72-hour forecast, finds the
peak AQI and when it first crosses "Unhealthy" (150) or "Very Unhealthy" (200),
and returns a severity + health advice. Surfaced as a banner in the dashboard;
the same structured output can drive email/webhook notifications.

## 15. CI/CD Automation

**Two GitHub Actions workflows (serverless, free):**
- **`feature-pipeline.yml`** — hourly (`cron: 5 * * * *`): fetch → engineer →
  store features for all cities.
- **`training-pipeline.yml`** — daily (`cron: 30 2 * * *`): retrain → batch
  inference → commit the refreshed `predictions.json` so the live dashboard updates.

**Details that matter:** secrets via GitHub Secrets (`HOPSWORKS_API_KEY`),
`concurrency` groups to prevent overlapping runs, `workflow_dispatch` for manual
triggers, and pip caching for speed. The green run history is the *proof* the
system is genuinely live and automated.

**Interview Q&A:**
- *Why is this "serverless"?* → No server to manage; GitHub runs the schedules,
  Hopsworks stores state, the frontend is static + a batch-written JSON.
- *What proves it actually runs?* → The Actions run history and the hourly/daily
  commits to the feature store and predictions file.

---

## 15b. MCP / LLM Air-Quality Advisor (differentiator)

**What it is:** a conversational AI assistant on the dashboard. A user asks
*"I have asthma — is it safe to jog in Lahore tomorrow?"* and the model answers
using our **real forecast**, not invented numbers.

**How it's grounded (the important part):** the LLM (Claude) is given **tools** —
`get_forecast(city)`, `get_history_summary(city)`, `list_cities()` — that call our
actual Python functions (`aqi/tools.py`). The model decides which tool to call,
reads the real AQI, and phrases health advice. This is **tool use / function
calling**, and it's what makes the answers trustworthy instead of hallucinated.

**Where MCP fits:** the *same* tool functions are also published over the **Model
Context Protocol** (`aqi/mcp/server.py`), so the system plugs into any MCP client
(e.g. Claude Desktop) — not just our own chat box. MCP is the standard "USB-C for
AI tools"; we expose our forecast system as a set of MCP tools.

**Architecture:**
```
 React ChatPanel ──▶ /api/chat (FastAPI) ──▶ advisor.py (Claude tool-use loop)
                                                   │ calls
                                                   ▼
                                            aqi/tools.py  ◀── also exposed via ──▶ MCP server
                                         (get_forecast, get_history, list_cities)
```

**Design choices to defend:**
- **Grounded, not generative:** every number comes from a tool call, so the model
  can't fabricate AQI values — the #1 risk with LLM apps.
- **Manual tool-use loop** (not a black-box agent framework): full control, no
  extra dependency; the loop runs tools until the model gives a final answer.
- **Degrades gracefully:** no `ANTHROPIC_API_KEY` → the endpoint returns a clean
  "advisor unavailable" message; the rest of the dashboard is unaffected.
- **One tool definition, two surfaces:** the advisor and the MCP server share the
  exact same functions (DRY) — the model can't drift from the real data.

**Interview Q&A:**
- *Isn't an LLM chatbot just a gimmick?* → Not when it's grounded in tools: it
  turns our model's output into plain-language health guidance, which is the
  actual user need. It never invents a number.
- *What is MCP?* → An open standard for connecting LLMs to tools/data uniformly;
  we expose our forecast system as MCP tools so it works in any MCP client.
- *How do you stop it from hallucinating AQI?* → It has no numbers of its own —
  it must call `get_forecast`, which returns our real model output.

## 16. General / Behavioral Questions

- **"Walk me through your project."** → Use the architecture diagram: four
  pipelines, feature store as the interface, all serverless.
- **"What was the hardest part?"** → (fill in a real one, e.g. the Windows
  Hopsworks build issue, or getting time-series validation right).
- **"What would you improve with more time?"** → Gas-pollutant unit conversion,
  more cities, probabilistic forecasts (prediction intervals), better DL tuning.
- **"How do you know your model is any good?"** → It beats a persistence baseline
  on walk-forward RMSE/MAE; metrics are logged in the Model Registry.
- **"How is this different from a Kaggle notebook?"** → It's automated,
  serverless, and live in production — it keeps working after I close my laptop.
- **"Why should we trust the predictions?"** → SHAP explains every one; alerts
  fire on hazardous days; the pipeline is observable via GitHub run history.

---

## Appendix A — Building the PDF

When the document is complete, convert Markdown → PDF with any of:

- **Pandoc** (best quality): `pandoc docs/interview-prep.md -o interview-prep.pdf`
  (needs a LaTeX engine like MiKTeX).
- **VS Code**: install the "Markdown PDF" extension → right-click → *Export (pdf)*.
- **Browser**: open a rendered Markdown preview → Print → *Save as PDF*.

---

*This document is a living artifact — updated after every module so it is
interview-ready and doubles as the source for the final project report.*
