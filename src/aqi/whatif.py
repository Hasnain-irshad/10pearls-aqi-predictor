"""What-If AQI simulator (#8).

Lets a user ask "what would the forecast be if the wind were stronger / PM2.5
lower / it rained?" — the model re-predicts with the overridden inputs and we
show baseline vs scenario. Clearly a **model simulation**, not causal proof.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.config import get_city
from aqi.data.openmeteo import fetch_raw
from aqi.features.engineering import build_features
from aqi.features.supervised import ANCHOR_STATE_FEATURES, TARGET_TIME_FEATURES
from aqi.models.registry import load_model
from aqi.utils.logging import get_logger

logger = get_logger("whatif")

# The sliders the UI exposes → the model features they map to.
SLIDERS = {
    "pm2_5": {"label": "Current PM2.5 (µg/m³)", "min": 0, "max": 400, "step": 5},
    "aqi_now": {"label": "Current AQI", "min": 0, "max": 500, "step": 5},
    "wind_speed": {"label": "Wind speed (km/h)", "min": 0, "max": 40, "step": 1},
    "humidity": {"label": "Humidity (%)", "min": 0, "max": 100, "step": 5},
    "temperature": {"label": "Temperature (°C)", "min": -5, "max": 50, "step": 1},
}


def _base_vector(city, bundle, horizon: int):
    """Build the model input for `city` at `horizon` hours ahead from live data."""
    raw = fetch_raw(city, past_days=3, forecast_days=2)
    feats = build_features(raw, dropna_target=False)
    now = pd.Timestamp.now(tz=city.timezone).tz_localize(None).floor("h")
    past = feats[feats["datetime"] <= now]
    anchor = past.iloc[-1] if len(past) else feats.iloc[0]
    future = feats[feats["datetime"] > now].reset_index(drop=True)
    target = future.iloc[min(horizon - 1, len(future) - 1)]

    row = {c: float(target[c]) for c in TARGET_TIME_FEATURES}
    for c in ANCHOR_STATE_FEATURES:
        row[f"{c}_anchor"] = float(anchor[c])
    row["horizon"] = horizon
    X = pd.DataFrame([row])[bundle.feature_columns].astype("float64")
    return X


def _apply(X: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """Map friendly slider values onto the model's feature columns."""
    X = X.copy()
    if (v := overrides.get("pm2_5")) is not None:
        X.loc[:, "pm2_5_anchor"] = float(v)
    if (v := overrides.get("aqi_now")) is not None:
        X.loc[:, "aqi_anchor"] = float(v)
        X.loc[:, "aqi_roll_mean_6h_anchor"] = float(v)
        X.loc[:, "aqi_roll_mean_24h_anchor"] = float(v)
    if (v := overrides.get("humidity")) is not None:
        X.loc[:, "relative_humidity_2m"] = float(v)
    if (v := overrides.get("temperature")) is not None:
        X.loc[:, "temperature_2m"] = float(v)
        X.loc[:, "apparent_temperature"] = float(v)
    if (v := overrides.get("wind_speed")) is not None:
        new = float(v)
        old = float(np.hypot(X["wind_u"].iloc[0], X["wind_v"].iloc[0])) or 1.0
        scale = new / old
        X.loc[:, "wind_u"] *= scale
        X.loc[:, "wind_v"] *= scale
        X.loc[:, "wind_speed_10m"] = new
        X.loc[:, "wind_gusts_10m"] = new * 1.4
    return X


def simulate(city_name: str, overrides: dict, horizon: int = 24) -> dict:
    """Return baseline vs scenario AQI for the given overrides."""
    from aqi.data.aqi import category_name

    city = get_city(city_name)
    bundle = load_model()
    X = _base_vector(city, bundle, horizon)
    base = float(bundle.estimator.predict(X)[0])
    scen = float(bundle.estimator.predict(_apply(X, overrides))[0])
    return {
        "city": city_name,
        "horizon_h": horizon,
        "baseline_aqi": round(base),
        "baseline_category": category_name(base),
        "scenario_aqi": round(scen),
        "scenario_category": category_name(scen),
        "delta": round(scen - base),
        "overrides": overrides,
    }


def slider_defaults(city_name: str, horizon: int = 24) -> dict:
    """Current real values for each slider, so the UI can start from reality."""
    city = get_city(city_name)
    bundle = load_model()
    X = _base_vector(city, bundle, horizon)
    r = X.iloc[0]
    return {
        "pm2_5": round(float(r["pm2_5_anchor"]), 1),
        "aqi_now": round(float(r["aqi_anchor"])),
        "wind_speed": round(float(np.hypot(r["wind_u"], r["wind_v"])), 1),
        "humidity": round(float(r["relative_humidity_2m"])),
        "temperature": round(float(r["temperature_2m"]), 1),
    }
