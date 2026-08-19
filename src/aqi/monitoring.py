"""Self-monitoring: data/feature drift + forecast-error tracking (#4, #15, #34, #35).

Turns a static model into a system that watches itself:

* **Drift** — compares recent feature distributions against a reference window
  using the Population Stability Index (PSI). If live data stops resembling the
  training data, that's a signal to retrain.
* **Forecast-error tracking** — each inference run logs its forecasts; once the
  real AQI for those times arrives, we join and measure how good the forecast
  actually was, and flag big misses for investigation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqi.config import PROCESSED_DIR, PROJECT_ROOT, ensure_dirs
from aqi.data.store import read_features
from aqi.utils.logging import get_logger

logger = get_logger("monitoring")

FORECAST_LOG = PROCESSED_DIR / "forecast_log.parquet"
DRIFT_REPORT = PROJECT_ROOT / "docs" / "monitoring.md"

DRIFT_FEATURES = [
    "aqi", "pm2_5", "pm10", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m", "surface_pressure",
]


def population_stability_index(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """PSI: <0.1 stable, 0.1–0.25 moderate drift, >0.25 significant drift."""
    ref, cur = ref.dropna(), cur.dropna()
    if len(ref) < 50 or len(cur) < 20:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    eps = 1e-6
    ref_pct, cur_pct = ref_pct + eps, cur_pct + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _status(psi: float) -> str:
    return "stable" if psi < 0.1 else ("moderate" if psi < 0.25 else "significant")


def drift_report(recent_days: int = 14) -> dict:
    """Compare the last `recent_days` of features vs everything before it."""
    df = read_features()
    df["dt"] = pd.to_datetime(df["datetime"])
    cutoff = df["dt"].max() - pd.Timedelta(days=recent_days)
    ref, cur = df[df["dt"] < cutoff], df[df["dt"] >= cutoff]

    features = []
    for f in DRIFT_FEATURES:
        if f in df.columns:
            psi = population_stability_index(ref[f], cur[f])
            features.append({"feature": f, "psi": round(psi, 3), "status": _status(psi)})
    worst = max(features, key=lambda x: x["psi"]) if features else {"psi": 0, "status": "stable"}
    report = {
        "recent_days": recent_days,
        "reference_rows": int(len(ref)),
        "recent_rows": int(len(cur)),
        "overall_status": worst["status"],
        "worst_feature": worst.get("feature"),
        "features": features,
    }
    _write_drift_md(report)
    logger.info("Drift: overall=%s (worst: %s PSI=%.3f)", report["overall_status"],
                worst.get("feature"), worst["psi"])
    return report


def log_forecasts(predictions: dict) -> None:
    """Append this run's hourly forecasts to the forecast log (for later scoring)."""
    ensure_dirs()
    gen = predictions.get("generated_at")
    rows = []
    for city, c in predictions.get("cities", {}).items():
        for h in c.get("hourly", []):
            rows.append({"generated_at": gen, "city": city,
                         "target_datetime": h["datetime"], "predicted_aqi": h["aqi"]})
    if not rows:
        return
    new = pd.DataFrame(rows)
    if FORECAST_LOG.exists():
        new = pd.concat([pd.read_parquet(FORECAST_LOG), new], ignore_index=True)
        new = new.drop_duplicates(["generated_at", "city", "target_datetime"], keep="last")
    new.to_parquet(FORECAST_LOG, index=False)
    logger.info("Logged %d forecast points (log now %d rows)", len(rows), len(new))


def forecast_error_report() -> dict:
    """Join logged forecasts with realized AQI to measure real forecast skill."""
    if not FORECAST_LOG.exists():
        return {"status": "no forecasts logged yet"}
    log = pd.read_parquet(FORECAST_LOG)
    actual = read_features()[["city", "datetime", "aqi"]].rename(
        columns={"datetime": "target_datetime", "aqi": "actual_aqi"})
    log["target_datetime"] = pd.to_datetime(log["target_datetime"])
    actual["target_datetime"] = pd.to_datetime(actual["target_datetime"])
    m = log.merge(actual, on=["city", "target_datetime"], how="inner")
    if m.empty:
        return {"status": "forecasts logged; waiting for actuals to arrive", "scored": 0}
    m["error"] = m["predicted_aqi"] - m["actual_aqi"]
    biggest = m.reindex(m["error"].abs().sort_values(ascending=False).index).head(5)
    return {
        "status": "ok",
        "scored_points": int(len(m)),
        "mae": round(float(m["error"].abs().mean()), 2),
        "rmse": round(float(np.sqrt((m["error"] ** 2).mean())), 2),
        "biggest_misses": [
            {"city": r.city, "when": str(r.target_datetime), "predicted": int(r.predicted_aqi),
             "actual": int(r.actual_aqi), "error": int(r.error)}
            for r in biggest.itertuples()
        ],
    }


def _write_drift_md(report: dict) -> None:
    lines = ["# Monitoring — Data Drift\n",
             f"Recent window: last {report['recent_days']} days "
             f"({report['recent_rows']:,} rows) vs reference ({report['reference_rows']:,} rows).\n",
             f"**Overall drift status: {report['overall_status'].upper()}**\n",
             "| Feature | PSI | Status |", "|---|---:|---|"]
    for f in report["features"]:
        lines.append(f"| {f['feature']} | {f['psi']} | {f['status']} |")
    lines.append("\n_PSI < 0.1 stable · 0.1–0.25 moderate · > 0.25 significant (retrain signal)._\n")
    DRIFT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(drift_report(), indent=2))
    print(json.dumps(forecast_error_report(), indent=2, default=str))
