import React from "react";
import ReactDOM from "react-dom/client";
import "leaflet/dist/leaflet.css";
import "./styles.css";
import App from "./App.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Remove the instant splash once the app has mounted.
const splash = document.getElementById("splash");
if (splash) {
  splash.style.transition = "opacity 0.4s ease";
  splash.style.opacity = "0";
  setTimeout(() => splash.remove(), 450);
}
