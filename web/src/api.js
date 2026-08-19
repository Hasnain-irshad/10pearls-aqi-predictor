// Tiny API client for the FastAPI backend.
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

export const api = {
  cities: () => get("/api/cities"),
  predictions: () => get("/api/predictions"),
  city: (name) => get(`/api/predictions/${encodeURIComponent(name)}`),
  categories: () => get("/api/categories"),
  chat: (question, history) => post("/api/chat", { question, history }),
  leaderboard: () => get("/api/leaderboard"),
  evaluation: () => get("/api/evaluation"),
  monitoring: () => get("/api/monitoring"),
  explain: (city) => get(`/api/explain/${encodeURIComponent(city)}`),
  whatifDefaults: (city) => get(`/api/whatif/defaults?city=${encodeURIComponent(city)}`),
  whatif: (city, overrides, horizon = 24) => post("/api/whatif", { city, overrides, horizon }),
};
