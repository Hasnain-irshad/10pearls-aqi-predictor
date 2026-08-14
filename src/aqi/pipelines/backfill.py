"""Backfill pipeline — one-off historical load for all cities.

Runs the feature logic over a long range of past dates to build the training
dataset, then writes it to the active store in one batch.

Open-Meteo's air-quality history (CAMS) starts ~2022-07-29, so that's the
earliest useful start date for AQI targets.

Usage:
    python -m aqi.pipelines.backfill --start 2023-08-01 --end 2025-07-01
    python -m aqi.pipelines.backfill --start 2024-01-01 --end 2024-03-01 --no-store
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta

import pandas as pd

from aqi.config import CITIES
from aqi.data.openmeteo import fetch_raw
from aqi.data.store import save_features
from aqi.features.engineering import build_features
from aqi.utils.logging import get_logger

logger = get_logger("backfill")

EARLIEST_AQ_HISTORY = "2022-08-01"


def _month_chunks(start: str, end: str, months: int = 3):
    """Yield (chunk_start, chunk_end) date-string pairs of ~`months` each."""
    cur = datetime.strptime(start, "%Y-%m-%d").date()
    stop = datetime.strptime(end, "%Y-%m-%d").date()
    while cur < stop:
        nxt = min(cur + timedelta(days=30 * months), stop)
        yield cur.isoformat(), nxt.isoformat()
        cur = nxt + timedelta(days=1)


def _backfill_city(city, start: str, end: str) -> pd.DataFrame:
    """Fetch a city's full history in chunks, then build features on the
    contiguous series (so lags/rollings are correct)."""
    frames = []
    for c_start, c_end in _month_chunks(start, end):
        frames.append(fetch_raw(city, start_date=c_start, end_date=c_end))
    raw = pd.concat(frames, ignore_index=True).drop_duplicates("datetime").sort_values("datetime")
    feats = build_features(raw)
    logger.info("%s: %d feature rows (AQI %.0f-%.0f)", city.name, len(feats),
                feats["aqi"].min(), feats["aqi"].max())
    return feats


def run(*, start: str = "2023-08-01", end: str | None = None, store: bool = True):
    end = end or (date.today() - timedelta(days=2)).isoformat()  # archive lags ~1-2 days
    logger.info("=== Backfill %s -> %s for %d cities ===", start, end, len(CITIES))

    all_feats = []
    for i, city in enumerate(CITIES, 1):
        logger.info("[%d/%d] Backfilling %s ...", i, len(CITIES), city.name)
        try:
            all_feats.append(_backfill_city(city, start, end))
        except Exception as exc:  # keep going if one city fails
            logger.error("  %s failed: %s", city.name, exc)

    combined = pd.concat(all_feats, ignore_index=True)
    logger.info("Backfill total: %d rows across %d cities", len(combined), len(all_feats))

    if store:
        save_features(combined)
    else:
        logger.info("--no-store: not writing to the store.")

    logger.info("=== Backfill done ===")
    return combined


def main() -> None:
    p = argparse.ArgumentParser(description="AQI historical backfill (all cities)")
    p.add_argument("--start", default="2023-08-01")
    p.add_argument("--end", default=None)
    p.add_argument("--no-store", action="store_true")
    args = p.parse_args()
    run(start=args.start, end=args.end, store=not args.no_store)


if __name__ == "__main__":
    main()
