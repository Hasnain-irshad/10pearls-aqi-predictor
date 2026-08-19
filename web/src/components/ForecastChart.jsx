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
          <defs>
            <linearGradient id="aqiFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#2dd4bf" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="aqiLine" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#22d3ee" />
              <stop offset="100%" stopColor="#2dd4bf" />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: "#8a97a8" }}
            interval={view === "hourly" ? 5 : 0}
          />
          <YAxis tick={{ fontSize: 11, fill: "#8a97a8" }} domain={[0, "auto"]} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={100} stroke="#ffde33" strokeDasharray="4 4" strokeOpacity={0.6} />
          <ReferenceLine y={150} stroke="#ff9933" strokeDasharray="4 4" strokeOpacity={0.6} />
          <ReferenceLine y={200} stroke="#ff5050" strokeDasharray="4 4" strokeOpacity={0.6} />
          <Area dataKey="range" stroke="none" fill="url(#aqiFill)" />
          <Line
            dataKey="aqi"
            stroke="url(#aqiLine)"
            strokeWidth={3}
            dot={view === "daily"}
            activeDot={{ r: 6, fill: "#22d3ee", stroke: "#fff", strokeWidth: 2 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
