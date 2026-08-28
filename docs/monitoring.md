# Monitoring — Data Drift

Recent window: last 14 days (5,217 rows) vs reference (283,383 rows).

**Overall drift status: SIGNIFICANT**

| Feature | PSI | Status |
|---|---:|---|
| aqi | 0.273 | significant |
| pm2_5 | 0.249 | moderate |
| pm10 | 0.103 | moderate |
| temperature_2m | 3.318 | significant |
| relative_humidity_2m | 0.316 | significant |
| wind_speed_10m | 0.051 | stable |
| surface_pressure | 0.774 | significant |

_PSI < 0.1 stable · 0.1–0.25 moderate · > 0.25 significant (retrain signal)._
