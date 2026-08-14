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

from aqi.config import CITIES, LOCATION, Location
from aqi.utils.logging import get_logger

logger = get_logger(__name__)

# Open-Meteo endpoints. Air quality handles both recent & historical in one URL.
# Weather has TWO: the forecast URL (recent + forecast) and the ERA5 archive URL
# (multi-year history) — we pick the archive for old date ranges (backfill).
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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

# Weather variables. Weather drives pollution dispersion — wind clears pollutants,
# temperature inversions trap them, rain washes them out — so these are among the
# most predictive features for AQI.
WEATHER_VARS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
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


def _fetch_hourly(
    url: str,
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    variables: Sequence[str],
    past_days: int | None = None,
    forecast_days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Shared helper: call an Open-Meteo endpoint and return a tidy hourly frame.

    Supports two modes on the same request shape:
    * recent/forecast: pass ``past_days`` / ``forecast_days``.
    * historical range: pass ``start_date`` / ``end_date`` (YYYY-MM-DD).
    """
    params: dict = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "hourly": ",".join(variables),  # API wants a comma-separated list
    }
    if start_date:
        params["start_date"] = start_date
        params["end_date"] = end_date or start_date
    else:
        params["past_days"] = past_days if past_days is not None else 7
        params["forecast_days"] = forecast_days if forecast_days is not None else 0

    response = _session().get(url, params=params, timeout=60)
    response.raise_for_status()  # turn any HTTP error into a clear exception
    payload = response.json()

    # Open-Meteo nests the data under "hourly", with a parallel "time" array.
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError(f"Unexpected Open-Meteo response from {url}: missing 'hourly.time'")

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])  # string -> real timestamps
    return df.sort_values("datetime").reset_index(drop=True)


def fetch_air_quality(
    city: Location = LOCATION,
    *,
    variables: Sequence[str] = AIR_QUALITY_VARS,
    past_days: int = 7,
    forecast_days: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch hourly air-quality data (pollutants + us_aqi) for a city.

    Use (start_date, end_date) for history, or (past_days, forecast_days) for
    recent + forecast. The air-quality endpoint serves both.
    """
    df = _fetch_hourly(
        AIR_QUALITY_URL,
        latitude=city.latitude, longitude=city.longitude, timezone=city.timezone,
        variables=variables, past_days=past_days, forecast_days=forecast_days,
        start_date=start_date, end_date=end_date,
    )
    logger.info("%s air-quality: %d rows (%s -> %s)", city.name, len(df), df["datetime"].min(), df["datetime"].max())
    return df


def fetch_weather(
    city: Location = LOCATION,
    *,
    variables: Sequence[str] = WEATHER_VARS,
    past_days: int = 7,
    forecast_days: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch hourly weather data for a city.

    Historical ranges (start_date given) hit the ERA5 *archive* endpoint;
    recent + forecast hit the forecast endpoint.
    """
    url = WEATHER_ARCHIVE_URL if start_date else WEATHER_URL
    df = _fetch_hourly(
        url,
        latitude=city.latitude, longitude=city.longitude, timezone=city.timezone,
        variables=variables, past_days=past_days, forecast_days=forecast_days,
        start_date=start_date, end_date=end_date,
    )
    logger.info("%s weather: %d rows (%s -> %s)", city.name, len(df), df["datetime"].min(), df["datetime"].max())
    return df


def fetch_raw(
    city: Location = LOCATION,
    *,
    past_days: int = 7,
    forecast_days: int = 0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch BOTH air quality and weather for ONE city and merge on ``datetime``.

    Returns one tidy hourly row per timestamp with every raw pollutant + weather
    variable, tagged with the city name and its coordinates (the coordinates
    become useful location features for the single global model).
    """
    aq = fetch_air_quality(city, past_days=past_days, forecast_days=forecast_days,
                           start_date=start_date, end_date=end_date)
    wx = fetch_weather(city, past_days=past_days, forecast_days=forecast_days,
                       start_date=start_date, end_date=end_date)
    merged = pd.merge(aq, wx, on="datetime", how="inner")
    merged.insert(1, "city", city.name)
    merged["latitude"] = city.latitude
    merged["longitude"] = city.longitude
    merged = merged.sort_values("datetime").reset_index(drop=True)
    logger.info("%s merged raw: %d rows x %d cols", city.name, merged.shape[0], merged.shape[1])
    return merged


def fetch_raw_all_cities(cities: Sequence[Location] = CITIES, *, past_days: int = 7, forecast_days: int = 0) -> pd.DataFrame:
    """Fetch merged raw data for EVERY supported city and stack into one frame."""
    frames = [fetch_raw(city, past_days=past_days, forecast_days=forecast_days) for city in cities]
    combined = pd.concat(frames, ignore_index=True)
    logger.info("All cities combined: %d rows across %d cities", len(combined), len(cities))
    return combined


if __name__ == "__main__":
    # Smoke test across two cities to prove the multi-city path works.
    from aqi.config import get_city

    for city in (get_city("Lahore"), get_city("Karachi")):
        frame = fetch_raw(city, past_days=2)
        latest = frame.iloc[-1]
        print(f"{city.name:12s} | {frame.shape[0]} rows x {frame.shape[1]} cols "
              f"| latest us_aqi={latest['us_aqi']} pm2_5={latest['pm2_5']}")
