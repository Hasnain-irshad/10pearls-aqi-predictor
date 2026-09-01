"""FastAPI backend for the Pearls AQI Predictor dashboard.

Serves the pre-computed forecasts (fast, serverless-friendly) and can also run
an on-demand forecast for ANY location via the loaded model.

Run locally:
    uvicorn aqi.api.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aqi.config import CITIES, MODELS_DIR, PROCESSED_DIR, Location
from aqi.data import published
from aqi.data.aqi import AQI_CATEGORIES
from aqi.pipelines.inference import PREDICTIONS_PATH, forecast_city

# The four documents the pipelines publish, as (repository path, bundled path).
# Reads go to the repository first so the API is as current as the last
# pipeline run, not as current as the last image build; see aqi.data.published.
_PREDICTIONS = ("data/processed/predictions.json", PREDICTIONS_PATH)
_EVALUATION = ("data/processed/evaluation.json", PROCESSED_DIR / "evaluation.json")
_MONITORING = ("data/processed/monitoring.json", PROCESSED_DIR / "monitoring.json")
_LEADERBOARD = ("models_local/leaderboard.json", MODELS_DIR / "leaderboard.json")

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
    data = published.load(*_PREDICTIONS)
    if not data:
        raise HTTPException(status_code=503, detail="No predictions available yet. Run the inference pipeline.")
    return data


@app.get("/api/health")
def health():
    """Liveness, plus how fresh the served forecast actually is.

    ``generated_at`` and ``forecast_source`` make staleness observable from
    outside: 'repository' means the document was read from the last pipeline
    run, 'bundled' means the copy baked into this image is being served because
    the repository could not be reached.
    """
    data = published.load(*_PREDICTIONS) or {}
    return {
        "status": "ok",
        "predictions_ready": bool(data),
        "generated_at": data.get("generated_at"),
        "forecast_source": published.source_of(_PREDICTIONS[0]),
    }


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


@app.get("/api/leaderboard")
def leaderboard():
    """Champion–Challenger model leaderboard."""
    from aqi.models.leaderboard import current_champion

    entries = published.load(*_LEADERBOARD, default=[]) or []
    return {"champion": current_champion(entries), "entries": entries}


@app.get("/api/evaluation")
def evaluation():
    """Per-horizon metrics + walk-forward backtest results."""
    data = published.load(*_EVALUATION)
    if not data:
        raise HTTPException(status_code=503, detail="Run the evaluation module first.")
    return data


@app.get("/api/monitoring")
def monitoring():
    """Data-drift status + forecast-error scoring.

    Prefers the published snapshot (written by the training pipeline) so a slim
    deployment serves it without a live Feature Store connection; falls back to
    computing live, then to a friendly placeholder so the tab never hard-fails.
    """
    snap = published.load(*_MONITORING)
    if snap:
        return snap
    try:
        from aqi.monitoring import drift_report, forecast_error_report

        return {"drift": drift_report(), "forecast_error": forecast_error_report()}
    except Exception:  # noqa: BLE001
        return {
            "drift": {"recent_days": 0, "overall_status": "stable", "worst_feature": None, "features": []},
            "forecast_error": {"status": "monitoring runs in the training pipeline; a snapshot will appear after the next run"},
        }


@app.get("/api/explain/{city}")
def explain(city: str):
    """Plain-language SHAP explanation of a city's forecast."""
    from aqi.tools import explain_prediction

    return explain_prediction(city)


@app.get("/api/whatif/defaults")
def whatif_defaults(city: str, horizon: int = 24):
    """Current real values for the What-If sliders."""
    from aqi.whatif import SLIDERS, slider_defaults

    try:
        return {"sliders": SLIDERS, "defaults": slider_defaults(city, horizon)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"What-If unavailable: {exc}")


class WhatIfRequest(BaseModel):
    city: str
    overrides: dict = {}
    horizon: int = 24


@app.post("/api/whatif")
def whatif(req: WhatIfRequest):
    """Re-predict AQI under overridden drivers (baseline vs scenario)."""
    from aqi.whatif import simulate

    try:
        return simulate(req.city, req.overrides, req.horizon)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Simulation failed: {exc}")


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/api/chat")
def chat(req: ChatRequest):
    """LLM air-quality advisor grounded in our forecasts.

    Works with whichever provider has a key configured - Gemini or Groq (both
    free) or Claude. See aqi.llm for the selection order.
    """
    from aqi.advisor import answer, status

    state = status()
    if not state["configured"]:
        raise HTTPException(status_code=503, detail=f"Advisor unavailable: {state['detail']}")

    try:
        return answer(req.question, req.history)
    except Exception as exc:  # surface a clean message to the UI
        raise HTTPException(status_code=502, detail=f"Advisor error: {exc}")


@app.get("/api/advisor")
def advisor_status():
    """Which advisor provider and model this deployment would use, if any."""
    from aqi.advisor import status

    return status()
