// Full-screen animated intro. Holds for a moment (and until data is ready),
// then `leaving` triggers the fade-out that reveals the dashboard.
export default function SplashScreen({ leaving }) {
  return (
    <div className={`splash-screen${leaving ? " leaving" : ""}`}>
      <div className="splash-inner">
        <img className="splash-logo" src="/10pearls-logo.png" alt="10Pearls" />
        <h2 className="splash-title">Pearls AQI Predictor</h2>
        <p className="splash-sub">3-day air quality forecasts across Pakistan</p>
        <div className="splash-dots" aria-hidden="true">
          <span style={{ background: "#00e400" }} />
          <span style={{ background: "#ffff00" }} />
          <span style={{ background: "#ff7e00" }} />
          <span style={{ background: "#ff0000" }} />
          <span style={{ background: "#8f3f97" }} />
        </div>
        <div className="splash-bar"><i /></div>
      </div>
    </div>
  );
}
