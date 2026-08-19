// Plain-language SHAP explanation of the selected city's forecast.
export default function ExplanationCard({ explanation }) {
  if (!explanation) return null;
  const maxAbs = Math.max(...explanation.contributors.map((c) => Math.abs(c.impact)), 1);
  return (
    <div className="explain-card">
      <h3>🧠 Why this forecast?</h3>
      <p className="explain-text">{explanation.text}</p>
      <div className="explain-bars">
        {explanation.contributors.map((c) => (
          <div key={c.feature} className="explain-row">
            <span className="explain-label">{c.label}</span>
            <div className="explain-track">
              <div
                className={`explain-bar ${c.impact >= 0 ? "pos" : "neg"}`}
                style={{ width: `${(Math.abs(c.impact) / maxAbs) * 100}%` }}
              />
            </div>
            <span className={`explain-val ${c.impact >= 0 ? "pos" : "neg"}`}>
              {c.impact >= 0 ? "+" : "−"}
              {Math.abs(c.impact)}
            </span>
          </div>
        ))}
      </div>
      <p className="explain-note">SHAP feature contributions (AQI points) for the peak forecast hour.</p>
    </div>
  );
}
