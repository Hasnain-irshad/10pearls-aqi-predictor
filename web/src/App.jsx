import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import Header from "./components/Header.jsx";
import CitySelect from "./components/CitySelect.jsx";
import CurrentCard from "./components/CurrentCard.jsx";
import AlertBanner from "./components/AlertBanner.jsx";
import ForecastChart from "./components/ForecastChart.jsx";
import PakistanMap from "./components/PakistanMap.jsx";
import Legend from "./components/Legend.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import ExplanationCard from "./components/ExplanationCard.jsx";
import ModelEvalPage from "./pages/ModelEvalPage.jsx";
import MonitoringPage from "./pages/MonitoringPage.jsx";
import WhatIfPage from "./pages/WhatIfPage.jsx";

const TABS = [
  { id: "forecast", label: "🌫️ Forecast" },
  { id: "eval", label: "🏆 Model Evaluation" },
  { id: "monitoring", label: "🌊 Monitoring" },
  { id: "whatif", label: "🎛️ What-If" },
];

export default function App() {
  const [data, setData] = useState(null);
  const [selected, setSelected] = useState("Lahore");
  const [view, setView] = useState("hourly");
  const [tab, setTab] = useState("forecast");
  const [error, setError] = useState(null);

  useEffect(() => {
    api.predictions()
      .then((d) => {
        setData(d);
        if (d.cities && !d.cities[selected]) setSelected(Object.keys(d.cities)[0]);
      })
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cityNames = useMemo(() => (data ? Object.keys(data.cities) : []), [data]);
  const city = data?.cities?.[selected];

  return (
    <div className="app">
      <Header generatedAt={data?.generated_at} />

      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      {error && (
        <div className="notice error">
          Couldn't reach the API ({error}). Start the backend:{" "}
          <code>uvicorn aqi.api.main:app --port 8000</code>.
        </div>
      )}
      {!data && !error && (
        <div className="app-loading">
          <img src="/10pearls-logo.png" alt="10Pearls" />
          <div className="spinner" />
          <div className="loading-text">Loading forecasts…</div>
        </div>
      )}

      {/* ---- Forecast tab ---- */}
      {data && tab === "forecast" && (
        <>
          <div className="controls">
            <CitySelect cities={data.cities} value={selected} onChange={setSelected} />
            <div className="toggle">
              <button className={view === "hourly" ? "active" : ""} onClick={() => setView("hourly")}>Hourly (72h)</button>
              <button className={view === "daily" ? "active" : ""} onClick={() => setView("daily")}>Daily (3d)</button>
            </div>
          </div>
          {city && (
            <div className="grid">
              <section className="col-main">
                {city.alert?.severity !== "none" && <AlertBanner alert={city.alert} />}
                <CurrentCard name={selected} city={city} />
                <ForecastChart city={city} view={view} />
                <ExplanationCard explanation={city.explanation} />
              </section>
              <aside className="col-side">
                <PakistanMap cities={data.cities} selected={selected} onSelect={setSelected} />
                <Legend />
                <ChatPanel city={selected} />
              </aside>
            </div>
          )}
        </>
      )}

      {/* ---- Other tabs ---- */}
      {data && tab === "eval" && <ModelEvalPage />}
      {data && tab === "monitoring" && <MonitoringPage />}
      {data && tab === "whatif" && (
        <WhatIfPage cities={data.cities} city={selected} onCity={setSelected} />
      )}

      <footer className="footer">
        <img className="footer-logo" src="/10pearls-logo.png" alt="10Pearls" />
        <div>
          Built for the 10Pearls Data Science Internship · {cityNames.length} cities · Data: Open-Meteo ·
          Champion model: global XGBoost · Feature store: Hopsworks
        </div>
      </footer>
    </div>
  );
}
