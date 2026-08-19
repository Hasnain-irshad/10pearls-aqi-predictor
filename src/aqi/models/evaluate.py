"""Rigorous evaluation: per-horizon error + walk-forward backtesting (Module: eval).

Two things serious forecasters do that most projects skip:

1. **Per-horizon metrics** — report accuracy separately for +1h, +24h, +72h, etc.
   Error grows with the forecast horizon; a single averaged number hides that.
   For a 3-day forecast this is the most important evaluation view.

2. **Walk-forward (rolling) backtesting** — instead of one train/test split, slide
   the split forward through time across several folds, always training on the
   past and testing on the future. This is the honest way to estimate how the
   model will do going forward, and it catches leakage a single split can miss.

Run:  python -m aqi.models.evaluate
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from aqi.config import PROJECT_ROOT
from aqi.data.store import read_features
from aqi.features.supervised import (
    ANCHOR_AQI_COLUMN,
    DEFAULT_HORIZONS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    make_supervised,
    time_split,
)
from aqi.models.registry import load_model
from aqi.utils.logging import get_logger

logger = get_logger("evaluate")

IMG = PROJECT_ROOT / "docs" / "images"
REPORT = PROJECT_ROOT / "docs" / "evaluation.md"


def _metrics(y, p):
    return {
        "rmse": float(root_mean_squared_error(y, p)),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
    }


def per_horizon_metrics(sup_valid: pd.DataFrame, model) -> pd.DataFrame:
    """RMSE/MAE/R² for the trained model, broken out by forecast horizon."""
    rows = []
    preds = model.predict(sup_valid[FEATURE_COLUMNS])
    baseline = sup_valid[ANCHOR_AQI_COLUMN].to_numpy()
    y = sup_valid[TARGET_COLUMN].to_numpy()
    for h in sorted(sup_valid["horizon"].unique()):
        mask = sup_valid["horizon"].to_numpy() == h
        m = _metrics(y[mask], preds[mask])
        b = _metrics(y[mask], baseline[mask])
        rows.append({"horizon_h": int(h), **m, "baseline_rmse": b["rmse"]})
    return pd.DataFrame(rows)


def walk_forward_backtest(sup: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Rolling-window backtest: train on past folds, test on the next."""
    sup = sup.sort_values("datetime").reset_index(drop=True)
    X, y = sup[FEATURE_COLUMNS], sup[TARGET_COLUMN]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for i, (tr, te) in enumerate(tscv.split(X), 1):
        model = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42,
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        m = _metrics(y.iloc[te], model.predict(X.iloc[te]))
        test_start = str(sup["datetime"].iloc[te[0]].date())
        rows.append({"fold": i, "train_rows": len(tr), "test_rows": len(te),
                     "test_from": test_start, **m})
        logger.info("Fold %d: test from %s | RMSE=%.2f", i, test_start, m["rmse"])
    return pd.DataFrame(rows)


def run(*, max_rows: int = 200_000, n_splits: int = 5):
    features = read_features()
    sup = make_supervised(features, horizons=DEFAULT_HORIZONS, max_rows=max_rows)
    _, valid = time_split(sup, valid_frac=0.2)
    bundle = load_model()

    ph = per_horizon_metrics(valid, bundle.estimator)
    wf = walk_forward_backtest(sup, n_splits=n_splits)

    # --- Figures
    IMG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ph["horizon_h"], ph["rmse"], "-o", label="XGBoost", color="#2b8cbe")
    ax.plot(ph["horizon_h"], ph["baseline_rmse"], "--o", label="Persistence baseline", color="#e34a33")
    ax.set(title="Forecast error grows with horizon", xlabel="forecast horizon (hours ahead)",
           ylabel="RMSE (AQI)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(IMG / "error_by_horizon.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(wf["fold"].astype(str), wf["rmse"], color="#756bb1")
    ax.set(title=f"Walk-forward backtest ({n_splits} folds)", xlabel="fold (later = more recent test window)",
           ylabel="RMSE (AQI)")
    for _, r in wf.iterrows():
        ax.text(r["fold"] - 1, r["rmse"], f"{r['rmse']:.1f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(IMG / "walk_forward_backtest.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    _write_report(ph, wf)
    logger.info("Per-horizon + walk-forward evaluation complete.")
    return ph, wf


def _write_report(ph: pd.DataFrame, wf: pd.DataFrame):
    lines = ["# Rigorous Evaluation — per-horizon & walk-forward\n",
             "## Error by forecast horizon\n",
             "How RMSE grows as we predict further ahead (vs the persistence baseline).\n",
             "![error by horizon](images/error_by_horizon.png)\n",
             "| Horizon (h) | RMSE | MAE | R² | Baseline RMSE |",
             "|---:|---:|---:|---:|---:|"]
    for _, r in ph.iterrows():
        lines.append(f"| {r['horizon_h']} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['r2']:.3f} | {r['baseline_rmse']:.2f} |")
    lines += ["\n## Walk-forward backtest\n",
              "Rolling-window validation — each fold trains on the past, tests on the next window.\n",
              "![walk-forward](images/walk_forward_backtest.png)\n",
              "| Fold | Train rows | Test rows | Test from | RMSE | MAE | R² |",
              "|---:|---:|---:|---|---:|---:|---:|"]
    for _, r in wf.iterrows():
        lines.append(f"| {r['fold']} | {r['train_rows']:,} | {r['test_rows']:,} | {r['test_from']} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['r2']:.3f} |")
    lines.append(f"\n**Mean backtest RMSE: {wf['rmse'].mean():.2f}** (± {wf['rmse'].std():.2f})\n")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", REPORT)


if __name__ == "__main__":
    run()
