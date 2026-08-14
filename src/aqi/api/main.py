"""FastAPI backend for the Pearls AQI Predictor dashboard.

Serves the pre-computed forecasts (fast, serverless-friendly) and can also run
an on-demand forecast for ANY location via the loaded model.

Run locally:
    uvicorn aqi.api.main:app --reload --port 8000
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from aqi.config import CITIES, Location
from aqi.data.aqi import AQI_CATEGORIES
from aqi.pipelines.inference import PREDICTIONS_PATH, forecast_city

app = FastAPI(title="Pearls AQI Predictor API", version="0.1.0")

# Allow the React dev server (and, in production, the deployed frontend) to call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None  # lazily-loaded model, only needed for on-demand predictions


def _get_bundle():
    global _bundle
    if _bundle is None:
        from aqi.models.registry import load_model

        _bundle = load_model()
    return _bundle


def _load_predictions() -> dict:
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(status_code=503, detail="No predictions available yet. Run the inference pipeline.")
    return json.loads(PREDICTIONS_PATH.read_text())


@app.get("/api/health")
def health():
    return {"status": "ok", "predictions_ready": PREDICTIONS_PATH.exists()}


@app.get("/api/categories")
def categories():
    """AQI category reference (for the legend/colour scale)."""
    return [
        {"name": c.name, "low": c.lo, "high": c.hi, "emoji": c.emoji, "advice": c.advice}
        for c in AQI_CATEGORIES
    ]


@app.get("/api/cities")
def cities():
    """The supported cities, grouped-friendly (name, province, coordinates)."""
    return [
        {"name": c.name, "province": c.province, "lat": c.latitude, "lon": c.longitude}
        for c in CITIES
    ]


@app.get("/api/predictions")
def predictions():
    """Full pre-computed forecast payload for every city."""
    return _load_predictions()


@app.get("/api/predictions/{city}")
def city_prediction(city: str):
    data = _load_predictions()
    if city not in data["cities"]:
        raise HTTPException(status_code=404, detail=f"No forecast for '{city}'.")
    return {"city": city, "generated_at": data["generated_at"], **data["cities"][city]}


@app.get("/api/predict")
def predict_on_demand(name: str, lat: float, lon: float, province: str = "Custom"):
    """Live forecast for ANY location (the 'any city in Pakistan' feature)."""
    loc = Location(name=name, latitude=lat, longitude=lon, province=province)
    try:
        result = forecast_city(loc, _get_bundle())
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    return {"city": name, **result}
