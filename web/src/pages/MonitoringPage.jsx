import { useEffect, useState } from "react";
import { api } from "../api";

const STATUS_COLOR = { stable: "#00e400", moderate: "#ff9933", significant: "#ff5050" };

// Self-monitoring: data drift (PSI) + forecast-error scoring.
export default function MonitoringPage() {
  const [mon, setMon] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.monitoring().then(setMon).catch((e) => setErr(e.message));
  }, []);

  if (err) return <div className="notice error">Couldn't load monitoring ({err}).</div>;
  if (!mon) return <div className="notice">Computing drift & error metrics…</div>;

  const { drift, forecast_error: fe } = mon;
  return (
    <div className="col-main">
      <div className="panel-card">
        <h3>🌊 Data Drift (PSI)</h3>
        <p className="muted-line">
          Recent {drift.recent_days} days vs the training history. PSI &lt; 0.1 stable · 0.1–0.25 moderate · &gt; 0.25 significant (retrain signal).
        </p>
        <div className="drift-status" style={{ borderColor: STATUS_COLOR[drift.overall_status] }}>
          Overall: <strong style={{ color: STATUS_COLOR[drift.overall_status] }}>{drift.overall_status.toUpperCase()}</strong>
          {drift.worst_feature && <> · worst driver: <strong>{drift.worst_feature}</strong></>}
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead><tr><th>Feature</th><th>PSI</th><th>Status</th></tr></thead>
            <tbody>
              {drift.features.map((f) => (
                <tr key={f.feature}>
                  <td>{f.feature}</td><td>{f.psi}</td>
                  <td><span className="pill" style={{ background: STATUS_COLOR[f.status] }}>{f.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel-card">
        <h3>🎯 Forecast Error Tracking</h3>
        {fe.status !== "ok" ? (
          <p className="muted-line">{fe.status}. As real AQI arrives for logged forecast times, accuracy is scored here automatically.</p>
        ) : (
          <>
            <p className="muted-line">
              Scored {Number(fe.scored_points).toLocaleString()} past forecasts against actuals — MAE {fe.mae}, RMSE {fe.rmse}.
            </p>
            <h4 style={{ margin: "10px 0 6px" }}>Biggest misses (auto-flagged for investigation)</h4>
            <div className="table-wrap">
              <table className="data-table">
                <thead><tr><th>City</th><th>When</th><th>Predicted</th><th>Actual</th><th>Error</th></tr></thead>
                <tbody>
                  {fe.biggest_misses.map((m, i) => (
                    <tr key={i}><td>{m.city}</td><td>{String(m.when).slice(0, 16)}</td>
                      <td>{m.predicted}</td><td>{m.actual}</td>
                      <td style={{ color: Math.abs(m.error) > 40 ? "#ff5050" : "#e6e9ee" }}>{m.error > 0 ? "+" : ""}{m.error}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
