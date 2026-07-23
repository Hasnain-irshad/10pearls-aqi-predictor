"""Central configuration for the Pearls AQI Predictor.

All tunable settings live here so pipelines, notebooks, and the app read
from a single source of truth. Secrets come from the environment (.env locally,
GitHub Actions secrets in CI) and are never hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (no-op in CI where secrets are injected as env vars).
load_dotenv()

# ---- Filesystem layout ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models_local"


def _get(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else val


@dataclass(frozen=True)
class Location:
    """The city we forecast AQI for."""

    name: str = _get("CITY_NAME", "Lahore")
    latitude: float = float(_get("LATITUDE", "31.5204"))
    longitude: float = float(_get("LONGITUDE", "74.3587"))
    timezone: str = _get("TIMEZONE", "Asia/Karachi")


@dataclass(frozen=True)
class HopsworksConfig:
    """Feature Store / Model Registry connection settings."""

    api_key: str | None = _get("HOPSWORKS_API_KEY")
    project: str = _get("HOPSWORKS_PROJECT", "aqi_predictor")

    # Feature Store object names (kept stable across the whole project).
    feature_group_name: str = "aqi_features"
    feature_group_version: int = 1
    feature_view_name: str = "aqi_feature_view"
    feature_view_version: int = 1
    model_name: str = "aqi_forecaster"


@dataclass(frozen=True)
class ForecastConfig:
    """Modeling horizon and target definition."""

    # Predict AQI for the next 3 days (72 hours ahead), matching the brief.
    horizon_hours: int = 72
    # We model a daily forecast; horizon in days for reporting.
    horizon_days: int = 3
    target_col: str = "aqi"


# Singletons imported elsewhere: `from aqi.config import LOCATION, HOPSWORKS, FORECAST`
LOCATION = Location()
HOPSWORKS = HopsworksConfig()
FORECAST = ForecastConfig()


def ensure_dirs() -> None:
    """Create local data/model directories if they don't exist."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
