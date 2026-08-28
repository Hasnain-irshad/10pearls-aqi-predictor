"""Local model registry: save/load a trained model bundle.

A "bundle" is everything inference needs: the fitted estimator, the exact feature
column order, the residual-quantile lookup for prediction intervals, and the
evaluation metrics/metadata. Saved to models_local/ as joblib + a human-readable
JSON. (When Hopsworks is configured we additionally push to its Model Registry.)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import joblib

from aqi.config import HOPSWORKS, MODELS_DIR, ensure_dirs
from aqi.utils.logging import get_logger

logger = get_logger(__name__)

BUNDLE_PATH = MODELS_DIR / "aqi_model.joblib"
META_PATH = MODELS_DIR / "aqi_model_meta.json"


def _use_registry() -> bool:
    """True where Hopsworks is configured AND installed (Linux/CI, deployed API)."""
    import importlib.util

    return bool(HOPSWORKS.api_key) and importlib.util.find_spec("hopsworks") is not None


def _champion_metrics(bundle: "ModelBundle") -> dict:
    """Pull the champion's flat {rmse, mae, r2} out of the bundle for the registry."""
    val = (bundle.metrics or {}).get("validation", {})
    best = (bundle.metrics or {}).get("best_model")
    m = val.get(best, {}) if best else {}
    return {k: float(m.get(k, 0.0)) for k in ("rmse", "mae", "r2")}


@dataclass
class ModelBundle:
    estimator: Any
    feature_columns: list[str]
    horizons: list[int]
    # residual quantiles per horizon for prediction intervals: {h: [q_low, q_high]}
    interval_by_horizon: dict[int, list[float]]
    metrics: dict[str, Any] = field(default_factory=dict)
    model_name: str = "unknown"
    trained_at: str = ""

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("estimator")  # not JSON-serialisable
        return d


def save_model(bundle: ModelBundle) -> None:
    # Always keep a local copy (fast, used within the same job + local dev)...
    ensure_dirs()
    joblib.dump(bundle, BUNDLE_PATH)
    META_PATH.write_text(json.dumps(bundle.meta(), indent=2, default=str))
    logger.info("Saved model '%s' -> %s", bundle.model_name, BUNDLE_PATH.name)
    # ...and push to the Hopsworks Model Registry so the champion is durable
    # (survives the ephemeral CI runner) and loadable by a deployed backend.
    if _use_registry():
        try:
            from aqi.models.hopsworks_registry import save_to_registry

            save_to_registry(bundle, _champion_metrics(bundle))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model Registry push failed (%s); local save still succeeded.", exc)


def load_model() -> ModelBundle:
    # Prefer the durable champion from the registry (a fresh runner or a deployed
    # backend has no local file); fall back to the local copy.
    if _use_registry():
        try:
            from aqi.models.hopsworks_registry import load_champion_bundle

            return load_champion_bundle()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load champion from registry (%s); trying local copy.", exc)
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"No trained model at {BUNDLE_PATH}. Run the training pipeline first.")
    return joblib.load(BUNDLE_PATH)
