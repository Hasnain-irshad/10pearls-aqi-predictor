"""Training pipeline (Module 4) — runs daily (via GitHub Actions).

Reads features from the store, builds the multi-horizon supervised set, trains
several model families, evaluates them honestly on a *future* validation window,
compares them against a persistence baseline, fits prediction intervals, and
saves the best model bundle.

Usage:
    python -m aqi.pipelines.training_pipeline
    python -m aqi.pipelines.training_pipeline --max-rows 150000 --no-save
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from aqi.data.store import read_features
from aqi.features.supervised import (
    ANCHOR_AQI_COLUMN,
    DEFAULT_HORIZONS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    make_supervised,
    time_split,
)
from aqi.models.registry import ModelBundle, save_model
from aqi.utils.logging import get_logger

logger = get_logger("training")


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _candidate_models() -> dict:
    """The model families we compare (statistical -> tree ensembles)."""
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=120, max_depth=16, min_samples_leaf=10,
            max_features="sqrt", n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            n_jobs=-1, random_state=42,
        ),
    }


def _format_table(results: dict[str, dict]) -> str:
    lines = [f"{'Model':22s} {'RMSE':>8s} {'MAE':>8s} {'R2':>7s}", "-" * 48]
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
        lines.append(f"{name:22s} {m['rmse']:8.2f} {m['mae']:8.2f} {m['r2']:7.3f}")
    return "\n".join(lines)


def run(*, max_rows: int = 200_000, horizons=DEFAULT_HORIZONS, save: bool = True):
    logger.info("=== Training pipeline start ===")
    features = read_features()
    logger.info("Loaded %d feature rows across %d cities", len(features), features["city"].nunique())

    sup = make_supervised(features, horizons=horizons, max_rows=max_rows)
    train, valid = time_split(sup, valid_frac=0.2)
    X_tr, y_tr = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_va, y_va = valid[FEATURE_COLUMNS], valid[TARGET_COLUMN]

    results: dict[str, dict] = {}

    # --- Baseline: persistence ("AQI in h hours = AQI now"). Models must beat it.
    results["Persistence (baseline)"] = _metrics(y_va, valid[ANCHOR_AQI_COLUMN])

    fitted = {}
    for name, model in _candidate_models().items():
        logger.info("Training %s ...", name)
        model.fit(X_tr, y_tr)
        results[name] = _metrics(y_va, model.predict(X_va))
        fitted[name] = model
        logger.info("  %s: RMSE=%.2f MAE=%.2f R2=%.3f", name, *results[name].values())

    best_name = min(fitted, key=lambda n: results[n]["rmse"])
    best = fitted[best_name]
    logger.info("Best model: %s (RMSE=%.2f)", best_name, results[best_name]["rmse"])

    # --- Prediction intervals: empirical residual quantiles per horizon (split-conformal style)
    valid = valid.copy()
    valid["pred"] = best.predict(X_va)
    valid["resid"] = valid[TARGET_COLUMN] - valid["pred"]
    interval_by_horizon = {
        int(h): [float(g["resid"].quantile(0.1)), float(g["resid"].quantile(0.9))]
        for h, g in valid.groupby("horizon")
    }

    table = _format_table(results)
    print("\n" + table + "\n")

    bundle = ModelBundle(
        estimator=best,
        feature_columns=FEATURE_COLUMNS,
        horizons=list(horizons),
        interval_by_horizon=interval_by_horizon,
        metrics={"validation": results, "best_model": best_name, "n_train": len(train), "n_valid": len(valid)},
        model_name=best_name,
        trained_at=datetime.now().isoformat(timespec="seconds"),
    )
    if save:
        save_model(bundle)
        _write_metrics_report(results, best_name, len(train), len(valid))

    logger.info("=== Training pipeline done ===")
    return bundle, results


def _write_metrics_report(results, best_name, n_train, n_valid) -> None:
    """Persist a metrics table for the report/README."""
    from aqi.config import PROJECT_ROOT

    path = PROJECT_ROOT / "docs" / "model_metrics.md"
    lines = [
        "# Model Evaluation (walk-forward validation)\n",
        f"- Train rows: {n_train:,} | Validation rows: {n_valid:,}",
        f"- **Best model: {best_name}**\n",
        "| Model | RMSE | MAE | R² |",
        "|-------|-----:|----:|---:|",
    ]
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
        lines.append(f"| {name} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['r2']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote metrics report -> %s", path)


def main() -> None:
    p = argparse.ArgumentParser(description="AQI training pipeline (global model)")
    p.add_argument("--max-rows", type=int, default=200_000)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    run(max_rows=args.max_rows, save=not args.no_save)


if __name__ == "__main__":
    main()
