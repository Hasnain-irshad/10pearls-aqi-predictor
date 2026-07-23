"""US EPA Air Quality Index (AQI) computation.

Open-Meteo already returns a ready-made ``us_aqi``, which we use as our primary
target. But we also compute the AQI from first principles here because:

1. It documents *exactly* what the number we predict means (great for the report
   and interview).
2. It powers the health-category labels and hazardous-level alerts (Module 8).
3. It lets us turn *predicted pollutant concentrations* into an AQI if we ever
   want to.

--------------------------------------------------------------------------------
THE MATH (know this cold for the interview)

AQI is not measured — it is *computed* from a pollutant concentration ``C`` using
a piecewise-linear transform. For the concentration bin ``[C_lo, C_hi]`` that a
reading falls into, and its matching index bin ``[I_lo, I_hi]``:

        AQI = (I_hi - I_lo) / (C_hi - C_lo) * (C - C_lo) + I_lo

Each pollutant produces its own *sub-index*. The overall AQI is the **maximum**
of the sub-indices — the single worst pollutant defines the air quality, because
health risk is driven by the worst offender, not an average.
--------------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The 6 AQI index bins (I_lo, I_hi) — identical for every pollutant.
_AQI_BINS: list[tuple[float, float]] = [
    (0, 50),      # Good
    (51, 100),    # Moderate
    (101, 150),   # Unhealthy for Sensitive Groups
    (151, 200),   # Unhealthy
    (201, 300),   # Very Unhealthy
    (301, 500),   # Hazardous
]

# Concentration breakpoints (C_lo, C_hi) per pollutant, index-aligned to _AQI_BINS.
#
# ⚠️ UNITS MATTER. PM2.5 / PM10 breakpoints are in µg/m³ — the SAME units
# Open-Meteo reports, so they can be used directly and are trustworthy.
#
# The gaseous pollutants are a trap: EPA defines their breakpoints in ppb/ppm,
# NOT µg/m³. A correct gas sub-index needs a molar-mass unit conversion first
# (e.g. CO ppm ↔ mg/m³). Because getting that exactly right (and matching EPA's
# averaging windows) is fiddly and PM2.5 dominates Lahore's AQI anyway, we
# compute the AQI from PM only (see POLLUTANTS_FOR_AQI below) and list proper
# gas handling as a documented future improvement. The gas tables are kept here
# for reference but are intentionally NOT used by default.
_CONC_BREAKS: dict[str, list[tuple[float, float]]] = {
    # --- Trusted (µg/m³, used) ---
    "pm2_5": [(0.0, 9.0), (9.1, 35.4), (35.5, 55.4), (55.5, 125.4), (125.5, 225.4), (225.5, 500.4)],
    "pm10": [(0, 54), (55, 154), (155, 254), (255, 354), (355, 424), (425, 604)],
    # --- Reference only (unit conversion needed; NOT used by default) ---
    "ozone": [(0, 108), (109, 140), (141, 170), (171, 210), (211, 400), (401, 600)],
    "carbon_monoxide": [(0.0, 5.2), (5.3, 10.8), (10.9, 14.3), (14.4, 17.6), (17.7, 34.8), (34.9, 57.5)],
    "sulphur_dioxide": [(0, 91), (92, 196), (197, 484), (485, 796), (797, 1583), (1584, 2630)],
    "nitrogen_dioxide": [(0, 100), (101, 188), (189, 677), (678, 1221), (1222, 2350), (2351, 3853)],
}

# Only these pollutants (units we trust) feed the AQI calculation.
POLLUTANTS_FOR_AQI: tuple[str, ...] = ("pm2_5", "pm10")


@dataclass(frozen=True)
class AQICategory:
    """One of the 6 EPA AQI health bands."""

    name: str
    lo: int
    hi: int
    emoji: str
    advice: str


# The 6 categories, used for labels, dashboard colours, and alerts (Module 8).
AQI_CATEGORIES: list[AQICategory] = [
    AQICategory("Good", 0, 50, "🟢", "Air quality is satisfactory; enjoy the outdoors."),
    AQICategory("Moderate", 51, 100, "🟡", "Unusually sensitive people should limit prolonged outdoor exertion."),
    AQICategory("Unhealthy for Sensitive Groups", 101, 150, "🟠", "Sensitive groups should reduce prolonged outdoor exertion."),
    AQICategory("Unhealthy", 151, 200, "🔴", "Everyone may feel effects; limit outdoor exertion."),
    AQICategory("Very Unhealthy", 201, 300, "🟣", "Health alert: avoid outdoor exertion; wear a mask outside."),
    AQICategory("Hazardous", 301, 500, "🟤", "Emergency: stay indoors and run air purifiers."),
]


def sub_index(concentration: float, breakpoints: list[tuple[float, float]]) -> float:
    """AQI sub-index for ONE pollutant reading via the piecewise-linear formula.

    Returns ``NaN`` for missing/invalid input (so it never crashes a batch),
    and caps at 500 for readings above the top breakpoint ("beyond the index").
    """
    if concentration is None or pd.isna(concentration) or concentration < 0:
        return np.nan
    for (c_lo, c_hi), (i_lo, i_hi) in zip(breakpoints, _AQI_BINS):
        if c_lo <= concentration <= c_hi:
            return (i_hi - i_lo) / (c_hi - c_lo) * (concentration - c_lo) + i_lo
    return 500.0  # above the highest breakpoint


def compute_sub_indices(df: pd.DataFrame, pollutants: tuple[str, ...] = POLLUTANTS_FOR_AQI) -> pd.DataFrame:
    """Per-pollutant AQI sub-indices for each row (columns like ``aqi_pm2_5``).

    Only ``pollutants`` with trusted units are included (PM2.5 + PM10 by default).
    """
    out = pd.DataFrame(index=df.index)
    for pollutant in pollutants:
        if pollutant in df.columns:
            breaks = _CONC_BREAKS[pollutant]
            out[f"aqi_{pollutant}"] = df[pollutant].apply(lambda c: sub_index(c, breaks))
    return out


def compute_aqi(df: pd.DataFrame) -> pd.Series:
    """Overall US AQI per row = the MAX sub-index across all pollutants."""
    sub = compute_sub_indices(df)
    return sub.max(axis=1).round()


def categorize(aqi_value: float) -> AQICategory:
    """Map an AQI number to its EPA health category."""
    if aqi_value is None or pd.isna(aqi_value):
        return AQI_CATEGORIES[0]
    for cat in AQI_CATEGORIES:
        if aqi_value <= cat.hi:
            return cat
    return AQI_CATEGORIES[-1]


def category_name(aqi_value: float) -> str:
    return categorize(aqi_value).name


def is_hazardous(aqi_value: float, threshold: int = 200) -> bool:
    """True when AQI is 'Very Unhealthy' or worse — the trigger for alerts."""
    return aqi_value is not None and not pd.isna(aqi_value) and aqi_value > threshold


if __name__ == "__main__":
    # Sanity checks + a live cross-check against Open-Meteo's own us_aqi.
    print("PM2.5 = 55.5 µg/m³ -> AQI", sub_index(55.5, _CONC_BREAKS["pm2_5"]), "(expected ~151)")
    print("PM2.5 = 9.0  µg/m³ -> AQI", sub_index(9.0, _CONC_BREAKS["pm2_5"]), "(expected 50)")
    print("PM2.5 = 20.0 µg/m³ -> AQI", round(sub_index(20.0, _CONC_BREAKS["pm2_5"]), 1))

    from aqi.data.openmeteo import fetch_air_quality

    df = fetch_air_quality(past_days=2)
    df["aqi_ours"] = compute_aqi(df)
    df["category"] = df["aqi_ours"].apply(category_name)
    comparison = df[["datetime", "pm2_5", "aqi_ours", "us_aqi", "category"]].tail(6)
    print("\nOur AQI vs Open-Meteo us_aqi (should be in the same ballpark):\n", comparison.to_string(index=False))
