"""AQI data tools — the grounded functions the LLM advisor and MCP server expose.

Both the in-dashboard advisor (``aqi.advisor``) and the standalone MCP server
(``aqi.mcp.server``) call THESE functions, so the model's answers are always
grounded in our real forecasts and history — never invented.
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from aqi.config import CITIES, CITIES_BY_NAME
from aqi.data.aqi import category_name
from aqi.pipelines.inference import PREDICTIONS_PATH


def list_cities() -> list[dict]:
    """All supported cities with their province and coordinates."""
    return [
        {"name": c.name, "province": c.province, "lat": c.latitude, "lon": c.longitude}
        for c in CITIES
    ]


def _load_predictions() -> dict:
    if not PREDICTIONS_PATH.exists():
        return {"cities": {}}
    return json.loads(PREDICTIONS_PATH.read_text())


def get_forecast(city: str) -> dict:
    """Current AQI + 3-day daily forecast + peak + alert for a city."""
    data = _load_predictions()
    if city not in data.get("cities", {}):
        return {"error": f"No forecast for '{city}'. Use list_cities to see options."}
    c = data["cities"][city]
    peak = max((h["aqi"] for h in c["hourly"]), default=c["current"]["aqi"])
    return {
        "city": city,
        "province": c["province"],
        "current_aqi": c["current"]["aqi"],
        "current_category": c["current"]["category"],
        "daily_forecast": c["daily"],          # [{date, aqi, lower, upper, category}, ...]
        "peak_aqi_next_3_days": peak,
        "peak_category": category_name(peak),
        "alert": c["alert"],
    }


def explain_prediction(city: str) -> dict:
    """Plain-language SHAP explanation of a city's forecast (why it's high/low)."""
    data = _load_predictions()
    if city not in data.get("cities", {}):
        return {"error": f"No forecast for '{city}'. Use list_cities to see options."}
    exp = data["cities"][city].get("explanation")
    if not exp:
        return {"error": "No explanation available (run the inference pipeline)."}
    return {
        "city": city,
        "explanation": exp["text"],
        "top_drivers": exp["contributors"],
        "for_time": exp.get("for_time"),
    }


@lru_cache(maxsize=1)
def _history() -> pd.DataFrame:
    """Load the historical feature table once (cached in-process)."""
    from aqi.data.store import read_features

    df = read_features()
    df["month"] = pd.to_datetime(df["datetime"]).dt.month
    return df


def get_history_summary(city: str) -> dict:
    """Long-run AQI statistics for a city (from the historical dataset)."""
    if city not in CITIES_BY_NAME:
        return {"error": f"Unknown city '{city}'."}
    df = _history()
    g = df[df["city"] == city]
    if g.empty:
        return {"error": f"No history for '{city}'."}
    monthly = g.groupby("month")["aqi"].mean()
    months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return {
        "city": city,
        "mean_aqi": round(float(g["aqi"].mean()), 1),
        "max_aqi_on_record": round(float(g["aqi"].max())),
        "worst_month": months[int(monthly.idxmax())],
        "cleanest_month": months[int(monthly.idxmin())],
        "hours_of_data": int(len(g)),
    }


# The tool set, described for the LLM (JSON-schema form the Anthropic API expects).
TOOL_SCHEMAS = [
    {
        "name": "list_cities",
        "description": "List all Pakistani cities the system can forecast, with province and coordinates. Call this if the user names a city you're unsure about.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_forecast",
        "description": "Get the current AQI and 3-day forecast (with a hazard alert) for a city. Use this for any question about upcoming air quality or whether it's safe to go outside.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. 'Lahore'"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_history_summary",
        "description": "Get long-run historical AQI statistics for a city (mean, record high, worst/cleanest month). Use this for questions about typical or seasonal air quality.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. 'Karachi'"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "explain_prediction",
        "description": "Explain WHY a city's forecast is high or low, in plain language, from the model's SHAP feature contributions. Use this when the user asks why the AQI is expected to change.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. 'Lahore'"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
]

# Dispatch table: tool name -> callable.
TOOL_IMPLS = {
    "list_cities": lambda **kw: list_cities(),
    "get_forecast": lambda city, **kw: get_forecast(city),
    "get_history_summary": lambda city, **kw: get_history_summary(city),
    "explain_prediction": lambda city, **kw: explain_prediction(city),
}


def run_tool(name: str, tool_input: dict):
    """Execute a tool by name with the given input dict."""
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return {"error": f"Unknown tool '{name}'."}
    return impl(**tool_input)
