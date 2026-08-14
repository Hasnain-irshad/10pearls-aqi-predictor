"""SHAP explainability (Module 6).

Explains WHY the model predicts what it does — globally (which features matter
most across all cities) and per-prediction (why tomorrow is bad for Lahore).
Produces a feature-importance figure for the report/dashboard.

SHAP (SHapley Additive exPlanations) assigns each feature a contribution to each
prediction, grounded in cooperative game theory (Shapley values). It's the gold
standard for model explanation and is required by the brief.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display (works in CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aqi.config import PROJECT_ROOT
from aqi.data.store import read_features
from aqi.features.supervised import FEATURE_COLUMNS, make_supervised
from aqi.models.registry import load_model
from aqi.utils.logging import get_logger

logger = get_logger("explain")

FIG_PATH = PROJECT_ROOT / "docs" / "images" / "shap_importance.png"


def global_importance(*, sample: int = 3000) -> pd.DataFrame:
    """Compute mean |SHAP| per feature on a sample; save a bar chart."""
    import shap

    bundle = load_model()
    feats = read_features()
    sup = make_supervised(feats, max_rows=sample * 4)
    X = sup[FEATURE_COLUMNS].sample(min(sample, len(sup)), random_state=42)

    estimator = bundle.estimator
    # Tree models (RF/XGBoost) -> fast exact TreeExplainer; else linear/kernel.
    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
    except Exception:
        explainer = shap.Explainer(estimator.predict, X)
        shap_values = explainer(X).values

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    top = imp.head(15)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2b8cbe")
    plt.xlabel("mean |SHAP value|  (impact on AQI prediction)")
    plt.title(f"Global feature importance — {bundle.model_name}")
    plt.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=130)
    plt.close()
    logger.info("Saved SHAP importance figure -> %s", FIG_PATH)

    print("\nTop 10 features by mean |SHAP|:")
    print(imp.head(10).to_string(index=False))
    return imp


if __name__ == "__main__":
    global_importance()
