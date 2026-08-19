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

# Human-friendly names so SHAP contributions read as plain English.
FRIENDLY = {
    "aqi_anchor": "current AQI", "pm2_5_anchor": "current PM2.5", "pm10_anchor": "current PM10",
    "aqi_roll_mean_24h_anchor": "24h AQI trend", "aqi_roll_mean_6h_anchor": "6h AQI trend",
    "aqi_roll_std_24h_anchor": "recent AQI volatility",
    "hour_sin": "time of day", "hour_cos": "time of day",
    "month_sin": "season", "month_cos": "season", "dow_sin": "day of week", "dow_cos": "day of week",
    "is_weekend": "weekend", "wind_u": "wind (E–W)", "wind_v": "wind (N–S)",
    "wind_speed_10m": "wind speed", "wind_gusts_10m": "wind gusts", "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity", "dew_point_2m": "dew point",
    "apparent_temperature": "feels-like temp", "precipitation": "rain",
    "surface_pressure": "air pressure", "cloud_cover": "cloud cover",
    "latitude": "location (lat)", "longitude": "location (lon)", "horizon": "lead time",
}


def friendly(name: str) -> str:
    return FRIENDLY.get(name, name.replace("_", " "))


_TREE_EXPLAINER = None


def _tree_explainer(estimator):
    global _TREE_EXPLAINER
    if _TREE_EXPLAINER is None:
        import shap

        _TREE_EXPLAINER = shap.TreeExplainer(estimator)
    return _TREE_EXPLAINER


def explain_row(x_row, bundle, top_k: int = 5) -> dict:
    """Explain a SINGLE forecast in plain language via SHAP.

    Returns the top feature contributions (signed, in AQI points) and a
    natural-language sentence, e.g. "current PM2.5 +31, low wind +18, rain −14".
    """
    if isinstance(x_row, pd.Series):
        X = x_row[bundle.feature_columns].to_frame().T
    elif isinstance(x_row, dict):
        X = pd.DataFrame([{c: x_row.get(c) for c in bundle.feature_columns}])
    else:
        X = pd.DataFrame(x_row, columns=bundle.feature_columns)
    X = X.astype("float64")

    explainer = _tree_explainer(bundle.estimator)
    shap_vals = np.asarray(explainer.shap_values(X))[0]
    base = float(np.ravel(explainer.expected_value)[0])
    pred = base + float(shap_vals.sum())

    order = np.argsort(np.abs(shap_vals))[::-1][:top_k]
    contributors = [
        {"feature": bundle.feature_columns[i], "label": friendly(bundle.feature_columns[i]),
         "impact": round(float(shap_vals[i]), 1)}
        for i in order
    ]
    parts = [f"{c['label']} {'+' if c['impact'] >= 0 else '−'}{abs(c['impact'])}" for c in contributors]
    text = f"Forecast AQI ≈ {round(pred)} (baseline {round(base)}). Main drivers: " + ", ".join(parts) + "."
    return {"prediction": round(pred), "base_value": round(base, 1), "contributors": contributors, "text": text}


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
