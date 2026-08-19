import { useEffect, useState } from "react";
import { api } from "../api";
import { categoryFor } from "../aqi";

// What-If simulator: override drivers, re-predict, compare baseline vs scenario.
export default function WhatIfPage({ cities, city, onCity }) {
  const [config, setConfig] = useState(null); // {sliders, defaults}
  const [values, setValues] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setConfig(null); setResult(null); setErr(null);
    api.whatifDefaults(city)
      .then((c) => { setConfig(c); setValues({ ...c.defaults }); })
      .catch((e) => setErr(e.message));
  }, [city]);

  async function run() {
    setBusy(true); setErr(null);
    try {
      setResult(await api.whatif(city, values, 24));
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  const cityNames = cities ? Object.keys(cities) : [];

  return (
    <div className="col-main">
      <div className="panel-card">
        <h3>🎛️ What-If Simulator</h3>
        <p className="muted-line">
          Change the drivers and see how the model's 24h AQI forecast for {city} responds. This is a model simulation, not causal proof.
        </p>

        <label className="city-select" style={{ marginBottom: 14 }}>
          <span>City</span>
          <select value={city} onChange={(e) => onCity(e.target.value)}>
            {cityNames.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>

        {err && <div className="notice error">{err}</div>}
        {!config && !err && <div className="notice">Loading current conditions…</div>}

        {config && values && (
          <>
            <div className="sliders">
              {Object.entries(config.sliders).map(([key, s]) => (
                <div key={key} className="slider-row">
                  <div className="slider-head"><span>{s.label}</span><b>{values[key]}</b></div>
                  <input type="range" min={s.min} max={s.max} step={s.step}
                    value={values[key]}
                    onChange={(e) => setValues((v) => ({ ...v, [key]: Number(e.target.value) }))} />
                </div>
              ))}
            </div>
            <button className="run-btn" onClick={run} disabled={busy}>
              {busy ? "Simulating…" : "Run simulation"}
            </button>
          </>
        )}
      </div>

      {result && (
        <div className="panel-card">
          <h3>Result</h3>
          <div className="scenario-compare">
            <ScenarioTile label="Baseline (current conditions)" aqi={result.baseline_aqi} cat={result.baseline_category} />
            <div className="scenario-arrow">→</div>
            <ScenarioTile label="Your scenario" aqi={result.scenario_aqi} cat={result.scenario_category} />
          </div>
          <p className="muted-line" style={{ textAlign: "center", marginTop: 10 }}>
            Change: <strong style={{ color: result.delta > 0 ? "#ff5050" : "#00e400" }}>
              {result.delta > 0 ? "+" : ""}{result.delta} AQI
            </strong> at +{result.horizon_h}h
          </p>
        </div>
      )}
    </div>
  );
}

function ScenarioTile({ label, aqi, cat }) {
  const c = categoryFor(aqi);
  return (
    <div className="scenario-tile">
      <div className="scenario-aqi" style={{ background: c.color, color: c.text }}>{aqi}</div>
      <div className="scenario-cat">{cat}</div>
      <div className="scenario-label">{label}</div>
    </div>
  );
}
