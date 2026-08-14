import { categoryFor, ADVICE } from "../aqi";

export default function CurrentCard({ name, city }) {
  const aqi = city.current.aqi;
  const cat = categoryFor(aqi);
  const peak = Math.max(...city.hourly.map((h) => h.aqi));
  const peakCat = categoryFor(peak);

  return (
    <div className="current-card" style={{ borderColor: cat.color }}>
      <div className="current-main" style={{ background: cat.color, color: cat.text }}>
        <div className="current-aqi">{aqi}</div>
        <div className="current-meta">
          <div className="current-city">{name}</div>
          <div className="current-cat">{cat.name}</div>
        </div>
      </div>
      <div className="current-body">
        <p className="advice">{ADVICE[cat.name]}</p>
        <div className="stat-row">
          <div className="stat">
            <span className="stat-label">Province</span>
            <span className="stat-val">{city.province}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Peak (next 3d)</span>
            <span className="stat-val" style={{ color: peakCat.color }}>
              {peak} · {peakCat.name}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
