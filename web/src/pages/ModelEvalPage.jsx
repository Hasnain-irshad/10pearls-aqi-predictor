import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid,
} from "recharts";
import { api } from "../api";

// Model Evaluation: champion–challenger leaderboard + per-horizon error + backtest.
export default function ModelEvalPage() {
  const [lb, setLb] = useState(null);
  const [ev, setEv] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.leaderboard().then(setLb).catch((e) => setErr(e.message));
    api.evaluation().then(setEv).catch(() => {});
  }, []);

  if (err) return <div className="notice error">Couldn't load model evaluation ({err}).</div>;

  return (
    <div className="col-main">
      {/* Leaderboard */}
      <div className="panel-card">
        <h3>🏆 Champion–Challenger Leaderboard</h3>
        <p className="muted-line">A new model is promoted only if it beats the current champion's RMSE.</p>
        {lb?.champion && (
          <div className="champion-badge">
            Champion: <strong>v{lb.champion.version} · {lb.champion.model}</strong> · RMSE {lb.champion.rmse} · R² {lb.champion.r2}
          </div>
        )}
        <div className="table-wrap">
          <table className="data-table">
            <thead><tr><th>Ver</th><th>Trained</th><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Result</th></tr></thead>
            <tbody>
              {(lb?.entries || []).slice().reverse().map((e) => (
                <tr key={e.version} className={e.is_champion ? "row-champ" : ""}>
                  <td>{e.version}</td><td>{String(e.trained_at).slice(0, 16)}</td><td>{e.model}</td>
                  <td>{e.rmse}</td><td>{e.mae}</td><td>{e.r2}</td>
                  <td>{e.is_champion ? "🏆 champion" : e.promoted ? "promoted" : "rejected"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-horizon error */}
      {ev && (
        <div className="panel-card">
          <h3>📉 Error by forecast horizon</h3>
          <p className="muted-line">RMSE grows the further ahead we predict — and beats the persistence baseline at every horizon.</p>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={ev.per_horizon} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
              <XAxis dataKey="horizon_h" tick={{ fontSize: 11, fill: "#9aa4b2" }}
                     label={{ value: "hours ahead", position: "insideBottom", offset: -2, fill: "#9aa4b2", fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11, fill: "#9aa4b2" }} />
              <Tooltip contentStyle={{ background: "#0b0e13", border: "1px solid #2a2f3a" }} />
              <Legend />
              <Line dataKey="rmse" name="XGBoost RMSE" stroke="#4aa3df" strokeWidth={2.5} />
              <Line dataKey="baseline_rmse" name="Baseline RMSE" stroke="#e34a33" strokeDasharray="5 4" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Walk-forward backtest */}
      {ev && (
        <div className="panel-card">
          <h3>🔁 Walk-forward backtest</h3>
          <p className="muted-line">
            Rolling validation across {ev.walk_forward.length} time folds — mean RMSE {ev.backtest_mean_rmse}.
          </p>
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>Fold</th><th>Test window from</th><th>Train rows</th><th>RMSE</th><th>R²</th></tr></thead>
              <tbody>
                {ev.walk_forward.map((f) => (
                  <tr key={f.fold}><td>{f.fold}</td><td>{f.test_from}</td>
                    <td>{Number(f.train_rows).toLocaleString()}</td><td>{Number(f.rmse).toFixed(2)}</td>
                    <td>{Number(f.r2).toFixed(3)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
