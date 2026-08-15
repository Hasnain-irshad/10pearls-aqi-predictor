"""Deep-learning forecaster (Module 5) — an LSTM sequence model.

Where the tree models treat each prediction as a tabular row, the LSTM consumes
an actual **time sequence**: the past `SEQ_LEN` hours of pollution + weather for a
city, and predicts the AQI `HORIZON` hours ahead. This is the "deep learning"
member of the model family the brief asks for.

We evaluate it *fairly* against XGBoost on the **same task** (predict AQI +24h),
with the same chronological train/validation split. On this kind of tabular,
weather-driven data the gradient-boosted model often wins — which is itself a
legitimate, defensible finding to report (don't use a neural net just because
it's fashionable; use what measures best).

Run (after TensorFlow is installed and the feature store is populated):
    python -m aqi.models.lstm
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from aqi.config import PROJECT_ROOT
from aqi.data.store import read_features
from aqi.utils.logging import get_logger

logger = get_logger("lstm")

SEQ_LEN = 48        # hours of history fed to the LSTM
HORIZON = 24        # predict AQI this many hours ahead
STRIDE = 6          # step between windows (keeps the dataset CPU-friendly)

SEQ_FEATURES = [
    "aqi", "pm2_5", "pm10",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "latitude", "longitude",
]


def build_sequences(features: pd.DataFrame):
    """Per-city sliding windows -> (X sequences, y target, window-end datetime)."""
    X_list, y_list, t_list = [], [], []
    for _, g in features.groupby("city", sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)
        feats = g[SEQ_FEATURES].to_numpy(dtype="float32")
        aqi = g["aqi"].to_numpy(dtype="float32")
        times = g["datetime"].to_numpy()
        last = len(g) - HORIZON
        for end in range(SEQ_LEN, last, STRIDE):
            window = feats[end - SEQ_LEN:end]
            if np.isnan(window).any():
                continue
            target = aqi[end + HORIZON - 1]
            if np.isnan(target):
                continue
            X_list.append(window)
            y_list.append(target)
            t_list.append(times[end])
    X = np.stack(X_list)
    y = np.array(y_list, dtype="float32")
    t = pd.to_datetime(np.array(t_list))
    logger.info("Built %d sequences of shape %s", len(X), X.shape[1:])
    return X, y, t


def _scale(X_train, X_valid):
    """Standardise features (fit on train only), preserving the 3-D shape."""
    n_feat = X_train.shape[2]
    scaler = StandardScaler().fit(X_train.reshape(-1, n_feat))
    tr = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    va = scaler.transform(X_valid.reshape(-1, n_feat)).reshape(X_valid.shape)
    return tr.astype("float32"), va.astype("float32")


def _metrics(y_true, y_pred):
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run(*, valid_frac: float = 0.2, epochs: int = 20):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(42)
    features = read_features()
    X, y, t = build_sequences(features)

    # Chronological split (train on the past, validate on the future).
    cutoff = pd.Series(t).quantile(1 - valid_frac)
    train_mask = t <= cutoff
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[~train_mask], y[~train_mask]
    X_tr, X_va = _scale(X_tr, X_va)
    logger.info("Train=%d  Valid=%d  (cutoff %s)", len(X_tr), len(X_va), cutoff)

    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN, len(SEQ_FEATURES))),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=epochs, batch_size=256, callbacks=[early], verbose=2)

    lstm_metrics = _metrics(y_va, model.predict(X_va, verbose=0).ravel())
    # Persistence baseline on the SAME task: AQI at the window end (unscaled).
    base_metrics = _metrics(y_va, _raw_persistence(X, y, t, cutoff))

    logger.info("LSTM  (+%dh): RMSE=%.2f MAE=%.2f R2=%.3f", HORIZON, *lstm_metrics.values())
    logger.info("Persist(+%dh): RMSE=%.2f MAE=%.2f R2=%.3f", HORIZON, *base_metrics.values())

    _write_report(lstm_metrics, base_metrics)
    return lstm_metrics, base_metrics


def _raw_persistence(X, y, t, cutoff):
    """Persistence prediction = AQI at window end (unscaled), for the valid split."""
    mask = t > cutoff
    return X[mask][:, -1, SEQ_FEATURES.index("aqi")]


def _write_report(lstm_metrics, base_metrics):
    path = PROJECT_ROOT / "docs" / "lstm_metrics.md"
    lines = [
        "# Deep Learning (LSTM) — Module 5\n",
        f"Task: predict AQI **{HORIZON}h ahead** from the past **{SEQ_LEN}h** "
        f"(global model, all cities, chronological split).\n",
        "| Model | RMSE | MAE | R² |", "|-------|-----:|----:|---:|",
        f"| LSTM | {lstm_metrics['rmse']:.2f} | {lstm_metrics['mae']:.2f} | {lstm_metrics['r2']:.3f} |",
        f"| Persistence (baseline) | {base_metrics['rmse']:.2f} | {base_metrics['mae']:.2f} | {base_metrics['r2']:.3f} |",
        "\n_Compare with the tabular models in `model_metrics.md`. Tree ensembles "
        "(XGBoost) tend to match or beat the LSTM on this weather-driven tabular "
        "data — a legitimate finding, not a failure._\n",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote LSTM report -> %s", path)


if __name__ == "__main__":
    run()
