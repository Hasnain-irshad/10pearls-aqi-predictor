# Monitoring — Data Drift

Recent window: last 14 days (7,414 rows) vs reference (689,546 rows).

**Overall drift status: SIGNIFICANT**

| Feature | PSI | Status |
|---|---:|---|
| aqi | 0.303 | significant |
| pm2_5 | 0.257 | significant |
| pm10 | 0.031 | stable |
| temperature_2m | 3.942 | significant |
| relative_humidity_2m | 0.393 | significant |
| wind_speed_10m | 0.014 | stable |
| surface_pressure | 1.253 | significant |

_PSI < 0.1 stable · 0.1–0.25 moderate · > 0.25 significant (retrain signal)._
