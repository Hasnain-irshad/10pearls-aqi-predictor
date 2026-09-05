// API client for the FastAPI backend — with a STATIC fallback.
// If VITE_API_URL is set, we talk to the live backend. If not, we run in
// "static mode": read the committed pipeline artifacts from /data/*.json so the
// dashboard works as a pure static site (no server). Features that need live
// compute (What-If, Chat) are hidden by the UI in static mode.
const BASE = import.meta.env.VITE_API_URL || "";
export const STATIC_MODE = !import.meta.env.VITE_API_URL;

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

async function getStatic(file) {
  const res = await fetch(`/data/${file}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function getWithStaticFallback(path, file) {
  try {
    return await get(path);
  } catch (error) {
    console.warn(`Falling back to bundled ${file}:`, error.message);
    return getStatic(file);
  }
}

const NO_BACKEND = () =>
  Promise.reject(new Error("This feature needs the live backend (not available in the static demo)."));

export const api = {
  predictions: () => (STATIC_MODE ? getStatic("predictions.json") : get("/api/predictions")),
  evaluation: () => (STATIC_MODE ? getStatic("evaluation.json") : get("/api/evaluation")),
  monitoring: () => (STATIC_MODE ? getStatic("monitoring.json") : get("/api/monitoring")),
  leaderboard: async () => {
    if (!STATIC_MODE) return get("/api/leaderboard");
    const entries = await getStatic("leaderboard.json");
    const champs = entries.filter((e) => e.is_champion);
    return { champion: champs[champs.length - 1] || null, entries };
  },
  categories: () => (STATIC_MODE ? getStatic("categories.json") : get("/api/categories")),
  // Analytics & SHAP — available in both modes (static JSON fallback).
  statistics: () => (STATIC_MODE ? getStatic("statistics.json") : getWithStaticFallback("/api/statistics", "statistics.json")),
  shapGlobal: () => (STATIC_MODE ? getStatic("shap_global.json") : getWithStaticFallback("/api/shap/global", "shap_global.json")),
  shapCity: (city) => (STATIC_MODE ? NO_BACKEND() : get(`/api/shap/${encodeURIComponent(city)}`)),
  // Live-compute features — unavailable in static mode (UI hides them).
  chat: (question, history) => (STATIC_MODE ? NO_BACKEND() : post("/api/chat", { question, history })),
  explain: (city) => (STATIC_MODE ? NO_BACKEND() : get(`/api/explain/${encodeURIComponent(city)}`)),
  whatifDefaults: (city) =>
    STATIC_MODE ? NO_BACKEND() : get(`/api/whatif/defaults?city=${encodeURIComponent(city)}`),
  whatif: (city, overrides, horizon = 24) =>
    STATIC_MODE ? NO_BACKEND() : post("/api/whatif", { city, overrides, horizon }),
};
