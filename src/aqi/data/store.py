"""Storage abstraction for features and predictions.

The whole pipeline talks to THIS module, never directly to a specific backend.
That means we can develop the entire system locally (Parquet files) today, and
the moment a Hopsworks API key is configured, the same code transparently writes
to the Hopsworks Feature Store instead — no downstream changes needed.

Backend selection:
* If ``HOPSWORKS_API_KEY`` is set (env/.env)  -> Hopsworks.
* Otherwise                                    -> local Parquet under data/processed/.
Force one explicitly with ``backend="local"`` or ``backend="hopsworks"``.
"""
from __future__ import annotations

import pandas as pd

from aqi.config import HOPSWORKS, PROCESSED_DIR, ensure_dirs
from aqi.utils.logging import get_logger

logger = get_logger(__name__)

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
PRIMARY_KEY = ["city", "timestamp"]


def _default_backend() -> str:
    return "hopsworks" if HOPSWORKS.api_key else "local"


def _hopsworks_importable() -> bool:
    """True only where the hopsworks SDK is installed (Linux/CI, not Windows)."""
    import importlib.util

    return importlib.util.find_spec("hopsworks") is not None


# ----------------------------- FEATURES ------------------------------------- #
def save_features(df: pd.DataFrame, *, backend: str | None = None) -> None:
    """Upsert a feature DataFrame into the active store (on the primary key)."""
    backend = backend or _default_backend()
    if backend == "hopsworks" and not _hopsworks_importable():
        logger.warning("hopsworks not installed here — falling back to local Parquet store.")
        backend = "local"
    if backend == "hopsworks":
        from aqi.data.hopsworks_store import insert_features, login

        insert_features(login(), df)
    else:
        _save_features_local(df)


def existing_cities(*, backend: str | None = None) -> set[str]:
    """Set of city names already stored (for resuming a partial backfill)."""
    backend = backend or _default_backend()
    if backend == "hopsworks" and not _hopsworks_importable():
        backend = "local"
    if backend == "hopsworks":
        from aqi.data.hopsworks_store import existing_cities as _hcities
        from aqi.data.hopsworks_store import login

        return _hcities(login())
    if FEATURES_PATH.exists():
        return set(pd.read_parquet(FEATURES_PATH, columns=["city"])["city"].dropna().unique())
    return set()


def read_features(*, backend: str | None = None) -> pd.DataFrame:
    """Read the full feature table back from the active store."""
    backend = backend or _default_backend()
    if backend == "hopsworks" and not _hopsworks_importable():
        logger.warning("hopsworks not installed here — reading from local Parquet store.")
        backend = "local"
    if backend == "hopsworks":
        from aqi.data.hopsworks_store import login
        from aqi.data.hopsworks_store import read_features as _hread

        return _hread(login())
    return _read_features_local()


def _save_features_local(df: pd.DataFrame) -> None:
    ensure_dirs()
    if FEATURES_PATH.exists():
        existing = pd.read_parquet(FEATURES_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=PRIMARY_KEY, keep="last")
    else:
        combined = df.copy()
    combined = combined.sort_values(PRIMARY_KEY).reset_index(drop=True)
    combined.to_parquet(FEATURES_PATH, index=False)
    logger.info("Saved %d rows -> %s (store now holds %d rows)", len(df), FEATURES_PATH.name, len(combined))


def _read_features_local() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"No local feature store at {FEATURES_PATH}. Run the backfill or feature pipeline first."
        )
    return pd.read_parquet(FEATURES_PATH).sort_values(PRIMARY_KEY).reset_index(drop=True)
