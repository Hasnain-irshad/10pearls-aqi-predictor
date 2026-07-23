"""Feature engineering for the AQI forecaster.

Turns the raw hourly weather + pollutant table (from ``aqi.data.openmeteo``)
into a model-ready feature table. Covers everything the brief asks for:

* **Time-based features** — hour, day, month, day-of-week, plus *cyclical*
  (sin/cos) encodings so the model understands that hour 23 is next to hour 0.
* **Derived features** — AQI change rate, lagged pollutants/AQI, rolling
  statistics, and wind decomposed into u/v vector components.

================================================================================
THE #1 THING TO UNDERSTAND HERE: TARGET LEAKAGE.

A lag/rolling feature must only ever look at the PAST. If a "feature" secretly
includes the current or a future value, the model sees the answer during
training, scores brilliantly in testing, and then fails in production. Every
rolling window below is ``.shift(1)``-ed so it excludes the current row. This
single detail is what separates a real ML engineer from a Kaggle copy-paster,
and interviewers love to probe it.
================================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.data.aqi import category_name, compute_aqi
from aqi.utils.logging import get_logger

logger = get_logger(__name__)

# Which columns we build lag & rolling features from.
_LAG_HOURS = (1, 3, 6, 12, 24)
_ROLL_WINDOWS = (6, 24)
_SERIES_COLS = ("aqi", "pm2_5", "pm10")


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"].dt
    df["hour"] = dt.hour
    df["day"] = dt.day
    df["month"] = dt.month
    df["day_of_week"] = dt.dayofweek
    df["day_of_year"] = dt.dayofyear
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)

    # Cyclical encodings so periodic time is represented on a circle, not a line.
    # hour 23 and hour 0 end up adjacent (distance ≈ 0) instead of far apart.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def _add_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
    # Wind direction is an angle (0–360°): 359° and 1° are almost the same wind,
    # but as raw numbers they look maximally different. Decomposing speed+direction
    # into u (east-west) and v (north-south) components fixes that and lets the
    # model reason about wind linearly.
    if {"wind_speed_10m", "wind_direction_10m"}.issubset(df.columns):
        rad = np.deg2rad(df["wind_direction_10m"])
        df["wind_u"] = df["wind_speed_10m"] * np.cos(rad)
        df["wind_v"] = df["wind_speed_10m"] * np.sin(rad)
    return df


def _add_change_rates(df: pd.DataFrame) -> pd.DataFrame:
    # "AQI change rate" — explicitly requested by the brief. Captures momentum:
    # is pollution rising or falling right now?
    df["aqi_change_1h"] = df["aqi"].diff(1)
    df["aqi_change_24h"] = df["aqi"].diff(24)
    prev = df["aqi"].shift(1)
    df["aqi_change_rate"] = (df["aqi"] - prev) / prev.replace(0, np.nan)  # guard /0
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # Air quality is autocorrelated: recent values predict the next one.
    for col in _SERIES_COLS:
        if col in df.columns:
            for lag in _LAG_HOURS:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # Rolling stats summarise the recent trend/volatility. The .shift(1) is the
    # anti-leakage guard: the window ends at the PREVIOUS hour, never includes now.
    for col in _SERIES_COLS:
        if col not in df.columns:
            continue
        for win in _ROLL_WINDOWS:
            roll = df[col].shift(1).rolling(win, min_periods=max(2, win // 2))
            df[f"{col}_roll_mean_{win}h"] = roll.mean()
            df[f"{col}_roll_std_{win}h"] = roll.std()
            df[f"{col}_roll_max_{win}h"] = roll.max()
    return df


def build_features(raw: pd.DataFrame, *, dropna_target: bool = True) -> pd.DataFrame:
    """Build the full feature table from a raw hourly DataFrame.

    Parameters
    ----------
    raw
        Output of ``aqi.data.openmeteo.fetch_raw`` (hourly, sorted by datetime).
    dropna_target
        Drop rows whose ``aqi`` target could not be computed.
    """
    df = raw.sort_values("datetime").reset_index(drop=True).copy()

    # --- Target: prefer Open-Meteo's (properly averaged) us_aqi; fall back to our
    #     own EPA calculation only where us_aqi is missing.
    computed = compute_aqi(df)
    if "us_aqi" in df.columns:
        df["aqi"] = df["us_aqi"].astype("float64")
        df["aqi"] = df["aqi"].fillna(computed)
    else:
        df["aqi"] = computed
    df["aqi_category"] = df["aqi"].apply(category_name)

    # --- Feature groups (order matters: change/lag/rolling need `aqi` to exist)
    df = _add_time_features(df)
    df = _add_wind_vectors(df)
    df = _add_change_rates(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)

    # --- Primary key for the feature store: epoch-seconds timestamp.
    #     (version-robust conversion that works on pandas 2.x and 3.x)
    df["timestamp"] = (df["datetime"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

    if dropna_target:
        df = df.dropna(subset=["aqi"]).reset_index(drop=True)

    logger.info("Built feature table: %d rows x %d cols", df.shape[0], df.shape[1])
    return df


if __name__ == "__main__":
    from aqi.data.openmeteo import fetch_raw

    raw = fetch_raw(past_days=5)
    feats = build_features(raw)
    print("\nRaw cols:", raw.shape[1], "-> Feature cols:", feats.shape[1])
    print("\nEngineered feature names:\n", [c for c in feats.columns if c not in raw.columns])
    show = ["datetime", "aqi", "aqi_category", "aqi_change_rate", "aqi_lag_24h", "aqi_roll_mean_24h", "wind_u"]
    print("\nSample (latest rows):\n", feats[show].tail(5).to_string(index=False))
    print("\nNaN count in key lag/rolling cols (expected: only at the very start):")
    print(feats[["aqi_lag_24h", "aqi_roll_mean_24h", "aqi_roll_std_24h"]].isna().sum())
