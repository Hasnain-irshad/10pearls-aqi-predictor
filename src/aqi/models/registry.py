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

from aqi.config import MODELS_DIR, ensure_dirs
from aqi.utils.logging import get_logger

logger = get_logger(__name__)

BUNDLE_PATH = MODELS_DIR / "aqi_model.joblib"
META_PATH = MODELS_DIR / "aqi_model_meta.json"


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
    ensure_dirs()
    joblib.dump(bundle, BUNDLE_PATH)
    META_PATH.write_text(json.dumps(bundle.meta(), indent=2, default=str))
    logger.info("Saved model '%s' -> %s", bundle.model_name, BUNDLE_PATH.name)


def load_model() -> ModelBundle:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"No trained model at {BUNDLE_PATH}. Run the training pipeline first.")
    return joblib.load(BUNDLE_PATH)
