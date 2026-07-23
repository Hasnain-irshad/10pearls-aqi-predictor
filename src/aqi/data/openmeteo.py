"""Fetch raw air-quality data for our city from the Open-Meteo API.

WHY Open-Meteo? It's free, needs **no API key**, and serves both *historical*
and *forecast* data — exactly what a 100% serverless pipeline needs. No secret
to manage means the GitHub Action that runs this hourly stays dead simple.

This module (Task 1.1) covers only the *fetch* step. Later tasks add AQI
computation, feature engineering, and storage.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from aqi.config import LOCATION
from aqi.utils.logging import get_logger

logger = get_logger(__name__)

# The Open-Meteo air-quality endpoint (separate from the weather endpoint).
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# The hourly pollutant variables we ask for. `us_aqi` is Open-Meteo's own
# pre-computed US AQI — handy to cross-check our own calculation later.
AIR_QUALITY_VARS: tuple[str, ...] = (
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
)


def _session(retries: int = 4, backoff: float = 1.5) -> requests.Session:
    """Build a requests session that automatically retries transient failures.

    WHY: this script runs unattended every hour in GitHub Actions. If the API
    returns a momentary 429/503, a naive `requests.get` would crash the whole
    run. `Retry` re-attempts with exponential backoff so a blip doesn't page us.
    """
    retry = Retry(
        total=retries,
        backoff_factor=backoff,  # waits 1.5s, 3s, 6s, ... between attempts
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_air_quality(
    *,
    latitude: float = LOCATION.latitude,
    longitude: float = LOCATION.longitude,
    timezone: str = LOCATION.timezone,
    variables: Sequence[str] = AIR_QUALITY_VARS,
    past_days: int = 7,
    forecast_days: int = 0,
) -> pd.DataFrame:
    """Fetch hourly air-quality data and return it as a tidy DataFrame.

    Parameters
    ----------
    latitude, longitude, timezone
        Location to query. Defaults come from ``aqi.config.LOCATION`` (Lahore),
        so we never hardcode coordinates in more than one place.
    past_days
        How many days of recent history to include (the hourly pipeline pulls a
        few so lag/rolling features have enough context downstream).
    forecast_days
        How many days of forecast to include (0 for now; used later for
        prediction inputs).

    Returns
    -------
    DataFrame with a ``datetime`` column and one column per requested variable.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "hourly": ",".join(variables),  # API wants a comma-separated list
        "past_days": past_days,
        "forecast_days": forecast_days,
    }

    logger.info("Fetching air quality for (%.4f, %.4f), past_days=%d", latitude, longitude, past_days)
    response = _session().get(AIR_QUALITY_URL, params=params, timeout=60)
    response.raise_for_status()  # turn any HTTP error into a clear exception
    payload = response.json()

    # Open-Meteo nests the data under "hourly", with a parallel "time" array.
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("Unexpected Open-Meteo response: missing 'hourly.time'")

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])  # string -> real timestamps
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info("Fetched %d rows (%s -> %s)", len(df), df["datetime"].min(), df["datetime"].max())
    return df


if __name__ == "__main__":
    # Quick manual smoke test: run `python -m aqi.data.openmeteo` and eyeball it.
    frame = fetch_air_quality(past_days=3)
    print("\nShape:", frame.shape)
    print("\nColumns:", list(frame.columns))
    print("\nHead:\n", frame.head())
    print("\nLatest us_aqi:", frame["us_aqi"].iloc[-1])
