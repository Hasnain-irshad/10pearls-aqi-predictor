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

export default function App() {
  const [data, setData] = useState(null); // full predictions payload
  const [selected, setSelected] = useState("Lahore");
  const [view, setView] = useState("hourly"); // "hourly" | "daily"
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .predictions()
      .then((d) => {
        setData(d);
        if (d.cities && !d.cities[selected]) {
          setSelected(Object.keys(d.cities)[0]);
        }
      })
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cityNames = useMemo(() => (data ? Object.keys(data.cities) : []), [data]);
  const city = data?.cities?.[selected];

  return (
    <div className="app">
      <Header generatedAt={data?.generated_at} />

      {error && (
        <div className="notice error">
          Couldn't reach the API ({error}). Start the backend:{" "}
          <code>uvicorn aqi.api.main:app --port 8000</code> and run the inference pipeline.
        </div>
      )}

      {!data && !error && <div className="notice">Loading forecasts…</div>}

      {data && (
        <>
          <div className="controls">
            <CitySelect
              cities={data.cities}
              value={selected}
              onChange={setSelected}
            />
            <div className="toggle">
              <button
                className={view === "hourly" ? "active" : ""}
                onClick={() => setView("hourly")}
              >
                Hourly (72h)
              </button>
              <button
                className={view === "daily" ? "active" : ""}
                onClick={() => setView("daily")}
              >
                Daily (3d)
              </button>
            </div>
          </div>

          {city && (
            <div className="grid">
              <section className="col-main">
                {city.alert?.severity !== "none" && <AlertBanner alert={city.alert} />}
                <CurrentCard name={selected} city={city} />
                <ForecastChart city={city} view={view} />
              </section>
              <aside className="col-side">
                <PakistanMap
                  cities={data.cities}
                  selected={selected}
                  onSelect={setSelected}
                />
                <Legend />
                <ChatPanel city={selected} />
              </aside>
            </div>
          )}
        </>
      )}

      <footer className="footer">
        Built for the 10Pearls Data Science Internship · Forecasts for {cityNames.length} cities ·
        Data: Open-Meteo · Model: global gradient-boosted forecaster
      </footer>
    </div>
  );
}
