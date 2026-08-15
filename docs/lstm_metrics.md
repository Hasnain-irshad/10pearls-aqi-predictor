# Deep Learning (LSTM) — Module 5

**Task:** predict AQI **24 hours ahead** from the past **48 hours** of pollution +
weather, as a single global model across all 22 cities, evaluated on a
chronological (future) validation split.

| Model (predict AQI +24h) | RMSE | MAE | R² |
|--------------------------|-----:|----:|---:|
| **LSTM** (2×LSTM + dense) | **22.12** | 14.32 | 0.799 |
| XGBoost (same +24h task)  | 22.63 | 14.48 | 0.789 |
| Persistence (baseline)    | 28.79 | 17.21 | 0.660 |

## Finding (honest and defensible)

- Both learned models **beat the persistence baseline** by a wide margin
  (~23% lower RMSE), confirming the forecast has real skill.
- At the **specific +24h horizon**, the **LSTM slightly edges out XGBoost**
  (22.12 vs 22.63 RMSE) — a sequence model can squeeze a little extra from the
  recent temporal pattern.
- **But XGBoost is the production model**, because:
  - It covers **all horizons 1–72h in one model** (overall RMSE 20.6), whereas
    this LSTM targets a single horizon.
  - It trains in ~1 minute vs. the LSTM's ~25 minutes on CPU.
  - It's easy to explain with SHAP and cheap to retrain daily.

**Takeaway for the interview:** we didn't pick a model because deep learning is
fashionable — we measured. The LSTM is a legitimate, competitive addition, but the
gradient-boosted model is the better engineering choice for this multi-horizon,
weather-driven, tabular problem.
