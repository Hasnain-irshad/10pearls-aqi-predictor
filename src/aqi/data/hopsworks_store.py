"""Hopsworks Feature Store / Model Registry backend.

Only imported when a Hopsworks API key is configured (see aqi.data.store). Keeps
all Hopsworks-specific calls in one place so the rest of the code stays backend
agnostic.
"""
from __future__ import annotations

import pandas as pd

from aqi.config import HOPSWORKS
from aqi.utils.logging import get_logger

logger = get_logger(__name__)

PRIMARY_KEY = ["city", "timestamp"]
EVENT_TIME = "datetime"


_PROJECT = None  # cached connection, reused across per-city inserts in the backfill


def login():
    """Authenticate and return the Hopsworks project handle (cached)."""
    global _PROJECT
    if _PROJECT is not None:
        return _PROJECT

    import hopsworks  # lazy import; only needed on the Hopsworks path

    if not HOPSWORKS.api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is not set (add it to .env or CI secrets).")
    logger.info("Logging in to Hopsworks project '%s'...", HOPSWORKS.project)
    _PROJECT = hopsworks.login(api_key_value=HOPSWORKS.api_key, project=HOPSWORKS.project)
    logger.info("Connected to Hopsworks (project id=%s)", _PROJECT.id)
    return _PROJECT


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable dtypes to plain numpy dtypes Hopsworks accepts."""
    out = df.copy()
    for col in out.columns:
        dtype = str(out[col].dtype)
        if dtype in ("Float64", "Float32", "Int64", "Int32", "boolean"):
            out[col] = out[col].astype("float64")
    return out


def get_feature_group(project):
    fs = project.get_feature_store()
    return fs.get_or_create_feature_group(
        name=HOPSWORKS.feature_group_name,
        version=HOPSWORKS.feature_group_version,
        primary_key=PRIMARY_KEY,
        event_time=EVENT_TIME,
        description="Hourly AQI features for Pakistani cities.",
        online_enabled=False,
        # Hopsworks 5.0 defaults new feature groups to DELTA, which needs a client
        # 'delta' library that the base pip install doesn't ship -> creation fails.
        # HUDI is materialised server-side (no client lib) and keeps primary-key
        # upserts (idempotent hourly inserts).
        time_travel_format="HUDI",
        # Disable statistics: the Deequ profiler over 74 columns OOMs the free-tier
        # Spark executor and fails the materialization job. Stats are cosmetic (UI
        # only); training reads the data, not the stats. Disabling = the job just
        # writes rows and finishes green.
        statistics_config=False,
    )


def insert_features(project, df: pd.DataFrame, *, wait: bool = False) -> None:
    """Upsert features into the feature group (non-blocking by default).

    We DON'T block on the offline materialization job (wait_for_job=False): on
    the free tier a Spark executor occasionally hangs at INITIALIZING, and a
    blocking insert then freezes the whole backfill until GitHub kills it at the
    6-hour cap. Non-blocking uploads the rows and lets Hopsworks materialise them
    server-side. This is safe because:
      * inserts are idempotent primary-key upserts (re-running never duplicates), and
      * the backfill resumes by skipping cities already present (see
        ``existing_cities``), so any city whose job didn't finish is simply redone.
    Pass wait=True only when a caller genuinely needs the rows queryable before it
    continues in the same process."""
    fg = get_feature_group(project)
    clean = _sanitize(df)
    logger.info("Inserting %d rows into feature group '%s' (wait_for_job=%s)...",
                len(clean), HOPSWORKS.feature_group_name, wait)
    fg.insert(clean, write_options={"wait_for_job": wait})
    logger.info("Insert submitted (materialization runs server-side).")


def existing_cities(project) -> set[str]:
    """Return the set of city names already present in the feature group.

    Lets the backfill resume without redoing finished cities. Reads only the
    ``city`` column (cheap) and is best-effort: on any error it returns an empty
    set so the caller falls back to processing every city (idempotent, just
    slower)."""
    fg = get_feature_group(project)
    try:
        df = fg.select(["city"]).read()
        cities = set(df["city"].dropna().unique())
        logger.info("Store already holds %d cities.", len(cities))
        return cities
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read existing cities (%s); will not skip any.", exc)
        return set()


def city_coverage(project) -> dict:
    """Return {city: earliest date present} in the feature group.

    Used for a *history-aware* resume: a city is only "done" if its data reaches
    back to the requested start date. This distinguishes a fully-backfilled city
    from one that merely has recent rows from the hourly feature pipeline. Reads
    only ``city`` + event-time columns and is best-effort (empty dict on error)."""
    fg = get_feature_group(project)
    try:
        df = fg.select(["city", EVENT_TIME]).read()
        df[EVENT_TIME] = pd.to_datetime(df[EVENT_TIME])
        mins = df.groupby("city")[EVENT_TIME].min()
        return {city: ts.date() for city, ts in mins.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read city coverage (%s); will not skip any.", exc)
        return {}


def get_feature_view(project):
    fs = project.get_feature_store()
    fg = get_feature_group(project)
    return fs.get_or_create_feature_view(
        name=HOPSWORKS.feature_view_name,
        version=HOPSWORKS.feature_view_version,
        query=fg.select_all(),
        description="All AQI features for training and inference.",
    )


def read_features(project) -> pd.DataFrame:
    fv = get_feature_view(project)
    df, _ = fv.training_data(description="full read")
    return df.sort_values(EVENT_TIME).reset_index(drop=True)
