// Plain-language SHAP explanation of the selected city's forecast with waterfall breakdown.
export default function ExplanationCard({ explanation }) {
  if (!explanation) return null;
  const maxAbs = Math.max(...explanation.contributors.map((c) => Math.abs(c.impact)), 1);
  const baseValue = explanation.base_value ?? (explanation.prediction - explanation.contributors.reduce((acc, c) => acc + c.impact, 0));

  return (
    <div className="explain-card">
      <div className="explain-header">
        <div className="explain-title-row">
          <h3>🧠 Why this forecast? (SHAP Explainability)</h3>
          <span className="explain-tag">Peak Hour Model Attribution</span>
        </div>
        <p className="explain-text">{explanation.text}</p>
      </div>

      {/* Baseline to Forecast Summary */}
      <div className="shap-waterfall-summary">
        <div className="shap-stat-pill">
          <span className="shap-stat-label">Model Baseline</span>
          <span className="shap-stat-val">{Math.round(baseValue)} AQI</span>
        </div>
        <span className="shap-arrow">➔</span>
        <div className="shap-stat-pill highlight">
          <span className="shap-stat-label">Forecast Result</span>
          <span className="shap-stat-val">{Math.round(explanation.prediction)} AQI</span>
        </div>
      </div>

      {/* Feature Contributions List */}
      <div className="explain-bars">
        {explanation.contributors.map((c) => {
          const isPos = c.impact >= 0;
          return (
            <div key={c.feature} className="explain-row">
              <div className="explain-label-group">
                <span className="explain-label">{c.label}</span>
                <span className={`impact-badge ${isPos ? "worsening" : "improving"}`}>
                  {isPos ? "↑ raises AQI" : "↓ lowers AQI"}
                </span>
              </div>
              <div className="explain-track">
                <div
                  className={`explain-bar ${isPos ? "pos" : "neg"}`}
                  style={{ width: `${(Math.abs(c.impact) / maxAbs) * 100}%` }}
                />
              </div>
              <span className={`explain-val ${isPos ? "pos" : "neg"}`}>
                {isPos ? "+" : "−"}
                {Math.abs(c.impact)}
              </span>
            </div>
          );
        })}
      </div>
      <p className="explain-note">
        SHAP (SHapley Additive exPlanations) values quantify exact AQI point contributions from weather, seasonality, and anchor pollution.
      </p>
    </div>
  );
}

