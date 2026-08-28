"""Hopsworks Model Registry backend: persist + load the champion model.

Only imported when hopsworks is installed and an API key is set. Makes the
trained champion **durable and versioned** — it survives the ephemeral CI
runner and can be loaded by a deployed backend — completing the Feature Store +
Model Registry MLOps stack. The champion is simply the registered version with
the lowest validation RMSE.
"""
from __future__ import annotations

import os
import tempfile

import joblib

from aqi.data.hopsworks_store import login
from aqi.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "aqi_global_forecaster"
_BUNDLE_FILE = "aqi_model.joblib"


def save_to_registry(bundle, metrics: dict) -> None:
    """Register the model bundle as a new version tagged with its metrics."""
    project = login()
    mr = project.get_model_registry()
    with tempfile.TemporaryDirectory() as d:
        joblib.dump(bundle, os.path.join(d, _BUNDLE_FILE))
        model = mr.python.create_model(
            name=MODEL_NAME,
            metrics=metrics,
            description="Global multi-horizon AQI forecaster (champion).",
        )
        model.save(d)
    logger.info("Registered '%s' v%s to Hopsworks Model Registry (rmse=%.3f).",
                MODEL_NAME, getattr(model, "version", "?"), metrics.get("rmse", float("nan")))


def champion_rmse() -> float | None:
    """Lowest RMSE currently in the registry (the champion), or None if empty."""
    try:
        mr = login().get_model_registry()
        m = mr.get_best_model(MODEL_NAME, "rmse", "min")
        return float(m.training_metrics["rmse"]) if m else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read champion from registry (%s).", exc)
        return None


def load_champion_bundle():
    """Download + load the best-RMSE (champion) model bundle from the registry."""
    mr = login().get_model_registry()
    m = mr.get_best_model(MODEL_NAME, "rmse", "min")
    if m is None:
        raise FileNotFoundError(f"No model '{MODEL_NAME}' in the Hopsworks Model Registry yet.")
    path = m.download()
    logger.info("Loaded champion '%s' v%s from Model Registry.", MODEL_NAME, getattr(m, "version", "?"))
    return joblib.load(os.path.join(path, _BUNDLE_FILE))
