"""Champion–Challenger model leaderboard + promotion gate (Module: MLOps #1).

Turns "we trained some models" into a real model lifecycle:

* Every training run logs its candidates (metrics, timestamp, version) to a
  persistent **leaderboard**.
* The best candidate is the **challenger**. It is **promoted to champion only if
  it beats the current champion** on validation RMSE. Otherwise the existing
  champion is kept — a new model never ships just because it's newer.

This is the MLOps discipline most student projects skip, and it's easy to demo:
"here's the leaderboard, here's why v7 was promoted and v8 was rejected."
"""
from __future__ import annotations

import json

from aqi.config import MODELS_DIR, PROJECT_ROOT, ensure_dirs
from aqi.utils.logging import get_logger

logger = get_logger("leaderboard")

LEADERBOARD_PATH = MODELS_DIR / "leaderboard.json"
LEADERBOARD_MD = PROJECT_ROOT / "docs" / "model_leaderboard.md"

# A challenger must beat the champion's RMSE by at least this margin to be
# promoted (guards against promoting on noise). 0 = any improvement promotes.
PROMOTION_MARGIN = 0.0


def load_leaderboard() -> list[dict]:
    if LEADERBOARD_PATH.exists():
        return json.loads(LEADERBOARD_PATH.read_text())
    return []


def current_champion(entries: list[dict] | None = None) -> dict | None:
    entries = entries if entries is not None else load_leaderboard()
    champs = [e for e in entries if e.get("is_champion")]
    return champs[-1] if champs else None


def record_run(candidates: dict[str, dict], best_name: str, meta: dict) -> dict:
    """Log a training run and run the promotion gate.

    Parameters
    ----------
    candidates : {model_name: {rmse, mae, r2}} for every model tried this run.
    best_name  : the best *fitted* candidate (the challenger).
    meta       : {trained_at, n_train, n_valid} run metadata.

    Returns the promotion decision.
    """
    entries = load_leaderboard()
    champion = current_champion(entries)
    challenger = candidates[best_name]
    version = max((e.get("version", 0) for e in entries), default=0) + 1

    promoted = champion is None or challenger["rmse"] < champion["rmse"] - PROMOTION_MARGIN
    if promoted:
        for e in entries:  # demote the old champion
            e["is_champion"] = False

    entry = {
        "version": version,
        "trained_at": meta.get("trained_at"),
        "model": best_name,
        "rmse": round(challenger["rmse"], 3),
        "mae": round(challenger["mae"], 3),
        "r2": round(challenger["r2"], 3),
        "n_train": meta.get("n_train"),
        "n_valid": meta.get("n_valid"),
        "candidates": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in candidates.items()},
        "is_champion": bool(promoted),
        "promoted": bool(promoted),
        "prev_champion_rmse": round(champion["rmse"], 3) if champion else None,
    }
    entries.append(entry)
    ensure_dirs()
    LEADERBOARD_PATH.write_text(json.dumps(entries, indent=2))
    _render_md(entries)

    decision = {
        "promoted": promoted,
        "version": version,
        "challenger_rmse": challenger["rmse"],
        "champion_rmse": champion["rmse"] if champion else None,
    }
    if promoted:
        logger.info("PROMOTED v%d (%s, RMSE %.2f)%s", version, best_name, challenger["rmse"],
                    "" if champion is None else f" — beat champion RMSE {champion['rmse']:.2f}")
    else:
        logger.info("NOT promoted: challenger RMSE %.2f did not beat champion %.2f; keeping champion",
                    challenger["rmse"], champion["rmse"])
    return decision


def _render_md(entries: list[dict]) -> None:
    champ = current_champion(entries)
    lines = ["# Model Leaderboard (Champion–Challenger)\n",
             f"**Current champion:** v{champ['version']} · {champ['model']} · "
             f"RMSE {champ['rmse']} (trained {champ['trained_at']})\n" if champ else "_No models yet._\n",
             "A challenger is promoted only if it beats the champion's validation RMSE.\n",
             "| Ver | Trained | Model | RMSE | MAE | R² | Result |",
             "|---:|---|---|---:|---:|---:|---|"]
    for e in reversed(entries):  # newest first
        tag = "🏆 champion" if e["is_champion"] else ("promoted" if e["promoted"] else "rejected")
        lines.append(f"| {e['version']} | {str(e['trained_at'])[:16]} | {e['model']} | "
                     f"{e['rmse']} | {e['mae']} | {e['r2']} | {tag} |")
    LEADERBOARD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
