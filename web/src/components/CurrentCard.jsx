import { categoryFor, ADVICE } from "../aqi";
import AqiGauge from "./AqiGauge.jsx";

export default function CurrentCard({ name, city }) {
  const aqi = city.current.aqi;
  const cat = categoryFor(aqi);
  const peak = Math.max(...city.hourly.map((h) => h.aqi));
  const peakCat = categoryFor(peak);

  return (
    <div className="current-card" style={{ borderColor: "rgba(255, 255, 255, 0.12)" }}>
      <div className="current-card-header">
        <div className="current-city-info">
          <span className="current-city-name">{name}</span>
          <span className="current-province-badge">{city.province}</span>
        </div>
        <div 
          className="current-category-pill" 
          style={{ background: cat.color, color: cat.text }}
        >
          {cat.name}
        </div>
      </div>

      <div className="gauge-wrapper">
        <AqiGauge value={aqi} size={250} />
      </div>

      <div className="current-body">
        <p className="advice">{ADVICE[cat.name]}</p>
        <div className="stat-row">
          <div className="stat">
            <span className="stat-label">Air Status</span>
            <span className="stat-val" style={{ color: cat.color }}>
              {cat.name}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">3-Day Peak</span>
            <span className="stat-val" style={{ color: peakCat.color }}>
              {peak} · {peakCat.name}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Coordinates</span>
            <span className="stat-val" style={{ fontSize: "13px", color: "var(--muted)" }}>
              {city.lat?.toFixed(2)}°N, {city.lon?.toFixed(2)}°E
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

