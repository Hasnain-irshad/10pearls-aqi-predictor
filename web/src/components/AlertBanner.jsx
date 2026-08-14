export default function AlertBanner({ alert }) {
  const isDanger = alert.severity === "danger";
  const when = alert.first_exceed_time
    ? new Date(alert.first_exceed_time).toLocaleString()
    : null;
  return (
    <div className={`alert-banner ${alert.severity}`}>
      <span className="alert-icon">{isDanger ? "🚨" : "⚠️"}</span>
      <div>
        <strong>
          {isDanger ? "Hazardous air expected" : "Unhealthy air expected"} — peak AQI{" "}
          {alert.peak_aqi} ({alert.category})
        </strong>
        <div className="alert-sub">
          {when ? `Crosses unhealthy around ${when}. ` : ""}
          {alert.advice}
        </div>
      </div>
    </div>
  );
}
