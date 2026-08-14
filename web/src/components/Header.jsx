export default function Header({ generatedAt }) {
  return (
    <header className="header">
      <div className="brand">
        <span className="logo">🌫️</span>
        <div>
          <h1>Pearls AQI Predictor</h1>
          <p className="tagline">3-day air quality forecasts across Pakistan</p>
        </div>
      </div>
      {generatedAt && (
        <div className="generated">
          Updated<br />
          <strong>{new Date(generatedAt).toLocaleString()}</strong>
        </div>
      )}
    </header>
  );
}
