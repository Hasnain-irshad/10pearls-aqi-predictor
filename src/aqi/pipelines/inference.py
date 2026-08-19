"""Batch inference — compute 3-day forecasts for all cities and write a JSON the
web app reads. Runs after each daily training (or on demand).

For each city we:
  1. Fetch recent history (for the anchor state) + the 3-day weather forecast.
  2. Anchor = the latest observed pollution state ("now").
  3. For each future hour tau (next 72h): features = forecasted weather/calendar/
     location at tau + anchor state + horizon; predict AQI(tau) + interval.
  4. Aggregate hourly -> daily; attach category + hazard alert.

Keeping this a *batch* job (not a live server) is what keeps the system
serverless: a scheduled Action writes predictions.json, the frontend just reads it.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from aqi.alerts import check_forecast
from aqi.config import CITIES, PROCESSED_DIR, ensure_dirs
from aqi.data.aqi import category_name
from aqi.data.openmeteo import fetch_raw
from aqi.features.engineering import build_features
from aqi.features.supervised import ANCHOR_STATE_FEATURES, TARGET_TIME_FEATURES
from aqi.models.registry import ModelBundle, load_model
from aqi.utils.logging import get_logger

logger = get_logger("inference")

PREDICTIONS_PATH = PROCESSED_DIR / "predictions.json"


def _nearest_horizon(h: int, trained: list[int]) -> int:
    return min(trained, key=lambda th: abs(th - h))


def _interval(pred: float, h: int, bundle: ModelBundle) -> tuple[float, float]:
    th = _nearest_horizon(h, list(bundle.interval_by_horizon.keys()))
    lo, hi = bundle.interval_by_horizon[th]
    return max(0.0, pred + lo), min(500.0, pred + hi)


def forecast_city(city, bundle: ModelBundle, *, horizon_hours: int = 72) -> dict:
    """Produce the hourly + daily forecast dict for one city."""
    raw = fetch_raw(city, past_days=3, forecast_days=4)  # 4 days ensures full 72h ahead
    feats = build_features(raw, dropna_target=False)

    now = pd.Timestamp.now(tz=city.timezone).tz_localize(None).floor("h")
    past = feats[feats["datetime"] <= now]
    anchor = past.iloc[-1] if len(past) else feats.iloc[0]
    future = feats[feats["datetime"] > now].head(horizon_hours).copy()
    if future.empty:
        raise RuntimeError(f"No future rows for {city.name}")

    # Build the model input matrix for every future hour.
    X = pd.DataFrame(index=future.index)
    for col in TARGET_TIME_FEATURES:
        X[col] = future[col].values
    for col in ANCHOR_STATE_FEATURES:
        X[f"{col}_anchor"] = anchor[col]
    X["horizon"] = ((future["datetime"] - now) / pd.Timedelta("1h")).round().astype(int).values
    X = X[bundle.feature_columns]

    preds = bundle.estimator.predict(X)
    hourly = []
    for (_, row), pred, h in zip(future.iterrows(), preds, X["horizon"].values):
        lo, hi = _interval(float(pred), int(h), bundle)
        hourly.append({
            "datetime": row["datetime"].isoformat(),
            "aqi": round(float(pred)),
            "lower": round(lo), "upper": round(hi),
            "category": category_name(float(pred)),
        })

    hdf = pd.DataFrame(hourly)
    hdf["date"] = pd.to_datetime(hdf["datetime"]).dt.date.astype(str)
    daily = [
        {
            "date": d,
            "aqi": round(g["aqi"].mean()),
            "lower": round(g["lower"].mean()), "upper": round(g["upper"].mean()),
            "category": category_name(g["aqi"].mean()),
        }
        for d, g in hdf.groupby("date")
    ][:3]

    alert = check_forecast(city.name, hdf.rename(columns={"aqi": "aqi"}), aqi_col="aqi", time_col="datetime")
    current_aqi = float(anchor["aqi"]) if pd.notna(anchor["aqi"]) else float(preds[0])

    # Plain-language SHAP explanation for the PEAK forecast hour (why it's high/low).
    explanation = None
    try:
        from aqi.models.explain import explain_row

        peak_i = int(np.argmax(preds))
        explanation = explain_row(X.iloc[peak_i], bundle)
        explanation["for_time"] = future.iloc[peak_i]["datetime"].isoformat()
    except Exception as exc:  # never let explainability break the forecast
        logger.warning("%s explanation skipped: %s", city.name, exc)

    return {
        "province": city.province,
        "lat": city.latitude, "lon": city.longitude,
        "current": {"aqi": round(current_aqi), "category": category_name(current_aqi)},
        "hourly": hourly,
        "daily": daily,
        "alert": alert.as_dict(),
        "explanation": explanation,
    }


def run(*, cities=CITIES, save: bool = True) -> dict:
    logger.info("=== Batch inference for %d cities ===", len(cities))
    bundle = load_model()
    logger.info("Loaded model '%s'", bundle.model_name)

    out = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"), "cities": {}}
    for city in cities:
        try:
            out["cities"][city.name] = forecast_city(city, bundle)
            peak = max(h["aqi"] for h in out["cities"][city.name]["hourly"])
            logger.info("  %s: peak 3-day AQI = %d", city.name, peak)
        except Exception as exc:
            logger.error("  %s failed: %s", city.name, exc)

    if save:
        ensure_dirs()
        PREDICTIONS_PATH.write_text(json.dumps(out, indent=2))
        logger.info("Wrote %d city forecasts -> %s", len(out["cities"]), PREDICTIONS_PATH)
        try:  # log forecasts so we can score them against actuals later (monitoring)
            from aqi.monitoring import log_forecasts

            log_forecasts(out)
        except Exception as exc:
            logger.warning("forecast logging skipped: %s", exc)
    logger.info("=== Inference done ===")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="AQI batch inference (all cities)")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    run(save=not args.no_save)


if __name__ == "__main__":
    main()
