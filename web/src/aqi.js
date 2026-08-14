// AQI category helpers — official US EPA colours + health guidance.

export const CATEGORIES = [
  { name: "Good", max: 50, color: "#00e400", text: "#04310a" },
  { name: "Moderate", max: 100, color: "#ffde33", text: "#3d3200" },
  { name: "Unhealthy for Sensitive Groups", max: 150, color: "#ff9933", text: "#3d1e00" },
  { name: "Unhealthy", max: 200, color: "#ff5050", text: "#3d0000" },
  { name: "Very Unhealthy", max: 300, color: "#b25aff", text: "#22003d" },
  { name: "Hazardous", max: 500, color: "#c81d3f", text: "#ffffff" },
];

export function categoryFor(aqi) {
  return CATEGORIES.find((c) => aqi <= c.max) ?? CATEGORIES[CATEGORIES.length - 1];
}

export function colorFor(aqi) {
  return categoryFor(aqi).color;
}

export const ADVICE = {
  Good: "Air quality is satisfactory — enjoy the outdoors.",
  Moderate: "Unusually sensitive people should limit prolonged outdoor exertion.",
  "Unhealthy for Sensitive Groups": "Sensitive groups should reduce prolonged outdoor exertion.",
  Unhealthy: "Everyone may feel effects — limit outdoor exertion.",
  "Very Unhealthy": "Health alert: avoid outdoor exertion; wear a mask outside.",
  Hazardous: "Emergency: stay indoors and run air purifiers.",
};
