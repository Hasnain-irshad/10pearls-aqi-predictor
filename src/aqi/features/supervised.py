"""Build the supervised (X, y) dataset for multi-horizon AQI forecasting.

FORECASTING DESIGN (important — be able to explain this):

We predict AQI at a future hour ``tau = t + h`` (h = 1..72). At prediction time
``t`` we know two kinds of information:

1. **Target-time features** — things we know about the future hour ``tau`` itself:
   the *forecasted* weather (Open-Meteo gives us this), the calendar (hour/month/
   day-of-week are deterministic), and the fixed location. These drive dispersion.

2. **Anchor-state features** — the latest *observed* pollution state at ``t``
   (current AQI, recent rolling means, current PM). These anchor the forecast to
   "where the air is right now". Obtained by shifting each series by ``h``.

The model learns ``AQI(tau) = f(target-time features, anchor state, horizon h)``.
Because ``horizon`` is itself a feature, ONE global model serves every city and
every lead time from +1h to +72h.

Assumption we document: at training time we use the *actual* weather at ``tau``;
at inference we use the *forecasted* weather. We assume the weather forecast is
reasonably accurate (Open-Meteo's is), which is standard practice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.utils.logging import get_logger

logger = get_logger(__name__)

# Features describing the FUTURE target hour tau (known via forecast/calendar).
TARGET_TIME_FEATURES: list[str] = [
    # weather at tau (from forecast)
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation", "surface_pressure", "cloud_cover",
    "wind_speed_10m", "wind_gusts_10m", "wind_u", "wind_v",
    # calendar at tau (deterministic)
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos", "is_weekend",
    # location (fixed per city)
    "latitude", "longitude",
]

# Observed pollution state at anchor time t (= tau - h); shifted by h at build.
ANCHOR_STATE_FEATURES: list[str] = [
    "aqi", "aqi_roll_mean_6h", "aqi_roll_mean_24h", "aqi_roll_std_24h", "pm2_5", "pm10",
]

# The full model input = target-time + anchor(_anchor suffix) + horizon.
FEATURE_COLUMNS: list[str] = (
    TARGET_TIME_FEATURES
    + [f"{c}_anchor" for c in ANCHOR_STATE_FEATURES]
    + ["horizon"]
)
TARGET_COLUMN = "y_aqi"
ANCHOR_AQI_COLUMN = "aqi_anchor"  # used by the persistence baseline

DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 36, 48, 60, 72)


def make_supervised(
    features: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    max_rows: int | None = 400_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Turn the hourly feature table into supervised rows across many horizons.

    For each city (processed independently so we never shift across a city
    boundary) and each horizon ``h``, we align the target ``AQI(tau)`` with the
    anchor state at ``tau - h``.
    """
    blocks: list[pd.DataFrame] = []
    for city, g in features.groupby("city", sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)
        for h in horizons:
            block = pd.DataFrame()
            # target-time features live on the target row tau (current row)
            for col in TARGET_TIME_FEATURES:
                block[col] = g[col]
            # anchor state comes from h hours earlier -> shift down by h
            for col in ANCHOR_STATE_FEATURES:
                block[f"{col}_anchor"] = g[col].shift(h)
            block["horizon"] = h
            block[TARGET_COLUMN] = g["aqi"]          # AQI at tau
            block["city"] = city
            block["datetime"] = g["datetime"]         # tau (for time-based split)
            blocks.append(block)

    sup = pd.concat(blocks, ignore_index=True)
    needed = [f"{c}_anchor" for c in ANCHOR_STATE_FEATURES] + [TARGET_COLUMN]
    sup = sup.dropna(subset=needed).reset_index(drop=True)

    if max_rows and len(sup) > max_rows:
        sup = sup.sample(max_rows, random_state=random_state).reset_index(drop=True)
        logger.info("Subsampled supervised set to %d rows", max_rows)

    logger.info("Supervised set: %d rows x %d features, horizons=%s",
                len(sup), len(FEATURE_COLUMNS), list(horizons))
    return sup


def time_split(sup: pd.DataFrame, *, valid_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation split (NEVER random for time series).

    The most recent ``valid_frac`` of time becomes validation, so we always test
    on the future relative to training — the only honest way to score a forecaster.
    """
    # Coerce to datetime locally so the split is correct regardless of how the
    # store returned the column (Hopsworks hands it back as strings).
    dt = pd.to_datetime(sup["datetime"])
    cutoff = dt.quantile(1 - valid_frac)
    train = sup[dt <= cutoff].reset_index(drop=True)
    valid = sup[dt > cutoff].reset_index(drop=True)
    logger.info("Time split at %s -> train=%d, valid=%d", cutoff, len(train), len(valid))
    return train, valid
