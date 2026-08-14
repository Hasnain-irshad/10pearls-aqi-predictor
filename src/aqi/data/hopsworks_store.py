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


def login():
    """Authenticate and return the Hopsworks project handle."""
    import hopsworks  # lazy import; only needed on the Hopsworks path

    if not HOPSWORKS.api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is not set (add it to .env or CI secrets).")
    logger.info("Logging in to Hopsworks project '%s'...", HOPSWORKS.project)
    project = hopsworks.login(api_key_value=HOPSWORKS.api_key, project=HOPSWORKS.project)
    logger.info("Connected to Hopsworks (project id=%s)", project.id)
    return project


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
    )


def insert_features(project, df: pd.DataFrame, *, wait: bool = True) -> None:
    fg = get_feature_group(project)
    clean = _sanitize(df)
    logger.info("Inserting %d rows into feature group '%s'...", len(clean), HOPSWORKS.feature_group_name)
    fg.insert(clean, write_options={"wait_for_job": wait})
    logger.info("Insert complete.")


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
