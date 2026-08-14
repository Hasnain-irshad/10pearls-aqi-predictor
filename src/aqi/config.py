"""Central configuration for the Pearls AQI Predictor.

All tunable settings live here so pipelines, notebooks, and the app read
from a single source of truth. Secrets come from the environment (.env locally,
GitHub Actions secrets in CI) and are never hard-coded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
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
    """A city we forecast AQI for."""

    name: str
    latitude: float
    longitude: float
    timezone: str = "Asia/Karachi"

    @property
    def slug(self) -> str:
        """Lowercase id used in keys/URLs, e.g. 'Lahore' -> 'lahore'."""
        return self.name.lower().replace(" ", "_")


# ---- The supported cities (single source of truth for the whole project) ----
# All major Pakistani cities share the Asia/Karachi timezone. Add/remove here and
# every pipeline, the model, and the dashboard dropdown pick it up automatically.
CITIES: list[Location] = [
    Location("Lahore", 31.5204, 74.3587),
    Location("Karachi", 24.8607, 67.0011),
    Location("Islamabad", 33.6844, 73.0479),
    Location("Rawalpindi", 33.5651, 73.0169),
    Location("Faisalabad", 31.4504, 73.1350),
    Location("Multan", 30.1575, 71.5249),
    Location("Peshawar", 34.0151, 71.5249),
    Location("Quetta", 30.1798, 66.9750),
    Location("Gujranwala", 32.1877, 74.1945),
    Location("Sialkot", 32.4945, 74.5229),
    Location("Hyderabad", 25.3960, 68.3578),
    Location("Bahawalpur", 29.3956, 71.6836),
]

CITIES_BY_NAME: dict[str, Location] = {c.name: c for c in CITIES}

# Default city (used when a single-city context needs one). Overridable via .env.
DEFAULT_CITY_NAME = _get("CITY_NAME", "Lahore")


def get_city(name: str) -> Location:
    """Look up a supported city by name (case-insensitive)."""
    for city in CITIES:
        if city.name.lower() == name.lower():
            return city
    raise KeyError(f"Unknown city '{name}'. Supported: {[c.name for c in CITIES]}")


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
LOCATION = get_city(DEFAULT_CITY_NAME)  # the default single city (Lahore)
HOPSWORKS = HopsworksConfig()
FORECAST = ForecastConfig()


def ensure_dirs() -> None:
    """Create local data/model directories if they don't exist."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
