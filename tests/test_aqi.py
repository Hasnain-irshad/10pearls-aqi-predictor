"""Unit tests for the US EPA AQI computation."""
import numpy as np
import pandas as pd

from aqi.data.aqi import (
    AQI_CATEGORIES,
    POLLUTANTS_FOR_AQI,
    categorize,
    compute_aqi,
    is_hazardous,
    sub_index,
    _CONC_BREAKS,
)


def test_sub_index_breakpoint_endpoint():
    # PM2.5 = 9.0 sits at the top of the "Good" bin -> AQI 50.
    assert round(sub_index(9.0, _CONC_BREAKS["pm2_5"])) == 50


def test_sub_index_monotonic():
    breaks = _CONC_BREAKS["pm2_5"]
    vals = [sub_index(c, breaks) for c in [1, 5, 10, 20, 40, 60]]
    assert vals == sorted(vals)


def test_compute_aqi_uses_only_trusted_pollutants():
    # CO is huge (µg/m³) but must NOT blow up the AQI (gases are excluded).
    df = pd.DataFrame({"pm2_5": [9.0], "pm10": [10], "carbon_monoxide": [3000]})
    assert float(compute_aqi(df).iloc[0]) == 50
    assert set(POLLUTANTS_FOR_AQI) == {"pm2_5", "pm10"}


def test_categorize_boundaries():
    assert categorize(50).name == "Good"
    assert categorize(51).name == "Moderate"
    assert categorize(301).name == "Hazardous"


def test_categories_contiguous():
    for prev, nxt in zip(AQI_CATEGORIES, AQI_CATEGORIES[1:]):
        assert nxt.lo == prev.hi + 1


def test_is_hazardous():
    assert is_hazardous(250) is True
    assert is_hazardous(150) is False


def test_nan_concentration_returns_nan():
    assert np.isnan(sub_index(np.nan, _CONC_BREAKS["pm2_5"]))
