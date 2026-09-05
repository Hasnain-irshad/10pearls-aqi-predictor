// Stage the committed pipeline artifacts into public/data so the static build
// can serve them with no backend. Runs before every build (npm "prebuild").
// Best-effort: if the source repo files aren't present (e.g. Vercel uploads
// only web/), it keeps whatever snapshot is already committed in public/data.
import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", ".."); // web/scripts -> web -> repo root
const out = resolve(here, "..", "public", "data");
mkdirSync(out, { recursive: true });

const files = [
  ["data/processed/predictions.json", "predictions.json"],
  ["data/processed/evaluation.json", "evaluation.json"],
  ["data/processed/monitoring.json", "monitoring.json"],
  ["models_local/leaderboard.json", "leaderboard.json"],
  ["data/processed/statistics.json", "statistics.json"],
  ["data/processed/shap_global.json", "shap_global.json"],
];

for (const [src, dst] of files) {
  const s = resolve(repo, src);
  if (existsSync(s)) {
    copyFileSync(s, resolve(out, dst));
    console.log("staged", dst);
  } else {
    console.warn("keep committed snapshot (source not found):", src);
  }
}
