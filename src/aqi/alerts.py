"""Hazardous-AQI alerting.

Scans a city's hourly forecast and raises an alert when the air is forecast to
become unhealthy, with the peak level, when it happens, and health advice. The
dashboard surfaces these; a notifier (email/webhook) can consume the same output.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aqi.data.aqi import categorize

# Alert thresholds map to EPA category boundaries.
WARNING_THRESHOLD = 150   # "Unhealthy" and above
DANGER_THRESHOLD = 200    # "Very Unhealthy" and above


@dataclass
class Alert:
    city: str
    severity: str            # "none" | "warning" | "danger"
    peak_aqi: float
    peak_time: str
    category: str
    advice: str
    first_exceed_time: str | None  # when it first crosses the warning threshold

    def as_dict(self) -> dict:
        return self.__dict__


def check_forecast(city: str, forecast: pd.DataFrame, *, aqi_col: str = "aqi", time_col: str = "datetime") -> Alert:
    """Build an Alert from an hourly forecast DataFrame (columns: datetime, aqi)."""
    peak_idx = forecast[aqi_col].idxmax()
    peak_aqi = float(forecast.loc[peak_idx, aqi_col])
    peak_time = str(forecast.loc[peak_idx, time_col])
    cat = categorize(peak_aqi)

    if peak_aqi >= DANGER_THRESHOLD:
        severity = "danger"
    elif peak_aqi >= WARNING_THRESHOLD:
        severity = "warning"
    else:
        severity = "none"

    exceed = forecast[forecast[aqi_col] >= WARNING_THRESHOLD]
    first_exceed = str(exceed[time_col].iloc[0]) if not exceed.empty else None

    return Alert(
        city=city,
        severity=severity,
        peak_aqi=round(peak_aqi),
        peak_time=peak_time,
        category=cat.name,
        advice=cat.advice,
        first_exceed_time=first_exceed,
    )
