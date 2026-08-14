// Tiny API client for the FastAPI backend.
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  cities: () => get("/api/cities"),
  predictions: () => get("/api/predictions"),
  city: (name) => get(`/api/predictions/${encodeURIComponent(name)}`),
  categories: () => get("/api/categories"),
};
