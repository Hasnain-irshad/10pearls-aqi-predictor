import { CATEGORIES } from "../aqi";

export default function Legend() {
  return (
    <div className="legend-card">
      <h3>AQI scale</h3>
      <ul className="legend">
        {CATEGORIES.map((c, i) => {
          const lo = i === 0 ? 0 : CATEGORIES[i - 1].max + 1;
          return (
            <li key={c.name}>
              <span className="swatch" style={{ background: c.color }} />
              <span className="legend-range">
                {lo}–{c.max === 500 ? "500+" : c.max}
              </span>
              <span className="legend-name">{c.name}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
