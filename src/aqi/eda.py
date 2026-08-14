"""Exploratory Data Analysis (Module 3).

Generates the key figures and a findings summary from the historical feature
store — the visual evidence that goes into the report and motivates the feature
choices. Run after the backfill:

    python -m aqi.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aqi.config import PROJECT_ROOT
from aqi.data.store import read_features
from aqi.utils.logging import get_logger

logger = get_logger("eda")

IMG_DIR = PROJECT_ROOT / "docs" / "images"
FINDINGS = PROJECT_ROOT / "docs" / "eda_findings.md"

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "cloud_cover", "wind_speed_10m",
]


def _save(fig, name: str):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    path = IMG_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path.name)


def run() -> dict:
    df = read_features()
    df["month"] = pd.to_datetime(df["datetime"]).dt.month
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    logger.info("EDA on %d rows, %d cities", len(df), df["city"].nunique())

    # 1) AQI distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["aqi"].dropna(), bins=60, color="#2b8cbe")
    ax.set(title="AQI distribution (all cities, full history)", xlabel="AQI", ylabel="hours")
    _save(fig, "eda_aqi_distribution.png")

    # 2) City ranking by mean AQI
    city_mean = df.groupby("city")["aqi"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(city_mean.index[::-1], city_mean.values[::-1], color="#e34a33")
    ax.set(title="Mean AQI by city", xlabel="mean AQI")
    _save(fig, "eda_city_ranking.png")

    # 3) Monthly seasonality (winter smog)
    monthly = df.groupby("month")["aqi"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly.values, "-o", color="#756bb1")
    ax.set(title="Seasonality — mean AQI by month", xlabel="month", ylabel="mean AQI",
           xticks=range(1, 13))
    _save(fig, "eda_seasonality.png")

    # 4) Diurnal pattern
    diurnal = df.groupby("hour")["aqi"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(diurnal.index, diurnal.values, "-o", color="#31a354")
    ax.set(title="Daily pattern — mean AQI by hour", xlabel="hour of day", ylabel="mean AQI",
           xticks=range(0, 24, 2))
    _save(fig, "eda_diurnal.png")

    # 5) Correlation of AQI with weather
    corr = df[["aqi"] + WEATHER_COLS].corr()["aqi"].drop("aqi").sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#3182bd" if v < 0 else "#de2d26" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.set(title="Correlation of weather with AQI", xlabel="Pearson r")
    ax.axvline(0, color="#333", lw=0.8)
    _save(fig, "eda_weather_correlation.png")

    # --- Findings summary
    worst_city = city_mean.index[0]
    cleanest_city = city_mean.index[-1]
    worst_month = int(monthly.idxmax())
    top_corr = corr.abs().sort_values(ascending=False).index[0]
    findings = {
        "rows": int(len(df)),
        "cities": int(df["city"].nunique()),
        "date_min": str(pd.to_datetime(df["datetime"]).min().date()),
        "date_max": str(pd.to_datetime(df["datetime"]).max().date()),
        "worst_city": worst_city,
        "worst_city_mean": round(float(city_mean.iloc[0]), 1),
        "cleanest_city": cleanest_city,
        "cleanest_city_mean": round(float(city_mean.iloc[-1]), 1),
        "worst_month": worst_month,
        "strongest_weather_corr": f"{top_corr} (r={corr[top_corr]:.2f})",
    }
    _write_findings(findings, city_mean, monthly)
    logger.info("EDA done: %s", findings)
    return findings


def _write_findings(f: dict, city_mean: pd.Series, monthly: pd.Series) -> None:
    months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    lines = [
        "# EDA — Key Findings\n",
        f"Dataset: **{f['rows']:,} hourly rows**, {f['cities']} cities, "
        f"{f['date_min']} → {f['date_max']}.\n",
        "## Highlights",
        f"- **Most polluted city:** {f['worst_city']} (mean AQI {f['worst_city_mean']}).",
        f"- **Cleanest city:** {f['cleanest_city']} (mean AQI {f['cleanest_city_mean']}).",
        f"- **Worst month:** {months[f['worst_month']]} — clear winter-smog seasonality.",
        f"- **Strongest weather driver:** {f['strongest_weather_corr']}.",
        "\n## Figures",
        "- `eda_aqi_distribution.png` — overall AQI distribution.",
        "- `eda_city_ranking.png` — mean AQI by city.",
        "- `eda_seasonality.png` — monthly seasonality (winter peak).",
        "- `eda_diurnal.png` — daily (hourly) pattern.",
        "- `eda_weather_correlation.png` — weather ↔ AQI correlations.",
        "\n## City ranking (mean AQI)\n",
        "| City | Mean AQI |", "|------|---------:|",
    ]
    for city, val in city_mean.items():
        lines.append(f"| {city} | {val:.1f} |")
    FINDINGS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote findings -> %s", FINDINGS)


if __name__ == "__main__":
    run()
