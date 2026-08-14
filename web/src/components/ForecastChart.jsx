import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
} from "recharts";
import { categoryFor } from "../aqi";

function fmtHour(iso) {
  const d = new Date(iso);
  return d.toLocaleString([], { weekday: "short", hour: "2-digit" });
}
function fmtDay(dateStr) {
  return new Date(dateStr).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const p = payload.find((x) => x.dataKey === "aqi");
  const aqi = p?.value;
  const cat = categoryFor(aqi);
  const d = payload[0]?.payload;
  return (
    <div className="tooltip">
      <div className="tooltip-time">{label}</div>
      <div className="tooltip-aqi" style={{ color: cat.color }}>
        AQI {aqi} · {cat.name}
      </div>
      {d && (
        <div className="tooltip-range">
          Range: {d.lower}–{d.upper}
        </div>
      )}
    </div>
  );
}

export default function ForecastChart({ city, view }) {
  const raw = view === "hourly" ? city.hourly : city.daily;
  const data = raw.map((d) => ({
    label: view === "hourly" ? fmtHour(d.datetime) : fmtDay(d.date),
    aqi: d.aqi,
    lower: d.lower,
    upper: d.upper,
    range: [d.lower, d.upper],
  }));

  return (
    <div className="chart-card">
      <div className="chart-head">
        <h3>{view === "hourly" ? "Next 72 hours" : "Next 3 days"}</h3>
        <span className="chart-hint">shaded band = 80% prediction interval</span>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: -12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: "#9aa4b2" }}
            interval={view === "hourly" ? 5 : 0}
          />
          <YAxis tick={{ fontSize: 11, fill: "#9aa4b2" }} domain={[0, "auto"]} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={100} stroke="#ffde33" strokeDasharray="4 4" />
          <ReferenceLine y={150} stroke="#ff9933" strokeDasharray="4 4" />
          <ReferenceLine y={200} stroke="#ff5050" strokeDasharray="4 4" />
          <Area dataKey="range" stroke="none" fill="#4aa3df" fillOpacity={0.18} />
          <Line
            dataKey="aqi"
            stroke="#4aa3df"
            strokeWidth={2.5}
            dot={view === "daily"}
            activeDot={{ r: 5 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
