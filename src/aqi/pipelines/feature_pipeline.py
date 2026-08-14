"""Feature pipeline — runs hourly (via GitHub Actions).

Fetches the latest weather + pollutant data for every supported city, engineers
features, and upserts them into the active store (local Parquet or Hopsworks).

Usage:
    python -m aqi.pipelines.feature_pipeline
    python -m aqi.pipelines.feature_pipeline --past-days 7 --no-store
"""
from __future__ import annotations

import argparse

from aqi.config import CITIES
from aqi.data.openmeteo import fetch_raw_all_cities
from aqi.data.store import save_features
from aqi.features.engineering import build_features_all_cities
from aqi.utils.logging import get_logger

logger = get_logger("feature_pipeline")


def run(*, past_days: int = 7, forecast_days: int = 0, store: bool = True):
    """Fetch recent data for all cities, build features, and (optionally) store.

    A few ``past_days`` are pulled so lag/rolling features for the newest hours
    have enough history. Re-running is safe: the store upserts on (city, timestamp).
    """
    logger.info("=== Feature pipeline start: %d cities, past_days=%d ===", len(CITIES), past_days)
    raw = fetch_raw_all_cities(past_days=past_days, forecast_days=forecast_days)
    features = build_features_all_cities(raw)

    if store:
        save_features(features)
    else:
        logger.info("--no-store: skipped writing (dry run).")

    logger.info("=== Feature pipeline done: %d rows ===", len(features))
    return features


def main() -> None:
    p = argparse.ArgumentParser(description="AQI hourly feature pipeline (all cities)")
    p.add_argument("--past-days", type=int, default=7)
    p.add_argument("--forecast-days", type=int, default=0)
    p.add_argument("--no-store", action="store_true")
    args = p.parse_args()
    run(past_days=args.past_days, forecast_days=args.forecast_days, store=not args.no_store)


if __name__ == "__main__":
    main()
