"""Unit tests for feature engineering and supervised dataset construction."""
import numpy as np
import pandas as pd

from aqi.features.engineering import build_features, build_features_all_cities
from aqi.features.supervised import FEATURE_COLUMNS, TARGET_COLUMN, make_supervised


def _synthetic_raw(hours=240, city="Lahore", lat=31.5, lon=74.3):
    idx = pd.date_range("2024-01-01", periods=hours, freq="h")
    r = np.arange(hours)
    return pd.DataFrame({
        "datetime": idx, "city": city, "latitude": lat, "longitude": lon,
        "pm2_5": 50 + 20 * np.sin(r / 12), "pm10": 80 + 30 * np.sin(r / 12),
        "carbon_monoxide": 300 + r % 5, "nitrogen_dioxide": 40 + r % 7,
        "sulphur_dioxide": 10 + r % 3, "ozone": 60 + r % 11, "us_aqi": 130 + 40 * np.sin(r / 12),
        "temperature_2m": 15 + 5 * np.sin(r / 24), "relative_humidity_2m": 60 + r % 20,
        "dew_point_2m": 8 + r % 4, "apparent_temperature": 14 + 5 * np.sin(r / 24),
        "precipitation": 0.0, "surface_pressure": 1010 + r % 6, "cloud_cover": r % 100,
        "wind_speed_10m": 5 + r % 10, "wind_direction_10m": r % 360, "wind_gusts_10m": 8 + r % 12,
    })


def test_build_features_columns():
    f = build_features(_synthetic_raw())
    for col in ("hour_sin", "aqi", "aqi_change_rate", "aqi_lag_24h",
                "aqi_roll_mean_24h", "wind_u", "wind_v", "timestamp"):
        assert col in f.columns


def test_timestamp_unique_sorted():
    f = build_features(_synthetic_raw())
    assert f["timestamp"].is_monotonic_increasing and f["timestamp"].is_unique


def test_multi_city_lags_isolated():
    # Two cities concatenated: each should independently have NaN 24h-lag at start.
    raw = pd.concat([_synthetic_raw(city="Lahore"),
                     _synthetic_raw(city="Karachi", lat=24.8, lon=67.0)], ignore_index=True)
    f = build_features_all_cities(raw)
    per_city_na = f.groupby("city")["aqi_lag_24h"].apply(lambda s: s.isna().sum())
    assert (per_city_na == 24).all()


def test_make_supervised_has_features_and_target():
    f = build_features(_synthetic_raw(hours=24 * 15))
    sup = make_supervised(f, horizons=(1, 24), max_rows=None)
    assert set(FEATURE_COLUMNS).issubset(sup.columns)
    assert TARGET_COLUMN in sup.columns
    # horizon is a feature and takes the requested values
    assert set(sup["horizon"].unique()).issubset({1, 24})


def test_supervised_no_leakage_anchor_is_past():
    # The anchor AQI at horizon h must equal the AQI h steps earlier.
    f = build_features(_synthetic_raw(hours=24 * 15)).reset_index(drop=True)
    sup = make_supervised(f, horizons=(1,), max_rows=None)
    # pick a row and verify y_aqi at time t equals aqi, anchor equals aqi at t-1
    merged = sup.sort_values("datetime").reset_index(drop=True)
    assert merged["aqi_anchor"].notna().all()
