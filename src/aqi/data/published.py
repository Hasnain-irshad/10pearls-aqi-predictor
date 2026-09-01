"""Runtime access to the artefacts the pipelines publish.

The training and inference pipelines commit ``predictions.json``,
``evaluation.json``, ``monitoring.json`` and ``leaderboard.json`` back to the
repository after every run. The deployed backend has to serve the *current*
ones, but its container image is built once and then frozen, so the copies baked
into the image go stale the moment the next scheduled run commits. That is
exactly what happened in production: the pipeline kept publishing fresh
forecasts while the API went on serving a three-day-old snapshot, because
nothing rebuilt the image.

This module closes that gap. It reads a published artefact from the repository
at request time, caches it in process for a short TTL, and falls back to the
copy bundled in the image whenever the remote read fails. The bundled copy
therefore becomes a *floor* on freshness rather than a ceiling: the API is as
current as the last pipeline run, and still serves something sensible when
GitHub is unreachable.

Configuration (environment variables):

``AQI_ARTIFACTS_BASE_URL``
    Base URL the artefacts are read from. Defaults to the project's own
    repository on the default branch. Set it to an empty string to disable
    remote reads entirely, which is what local development and the pipelines
    themselves want.
``AQI_ARTIFACTS_TTL``
    Seconds a fetched artefact is reused before being re-read. Default 300.
``AQI_ARTIFACTS_TIMEOUT``
    Per-request timeout in seconds. Default 6.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from aqi.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = (
    "https://raw.githubusercontent.com/Hasnain-irshad/10pearls-aqi-predictor/main/"
)


def _base_url() -> str:
    """Remote base URL, or an empty string when remote reads are disabled."""
    raw = os.getenv("AQI_ARTIFACTS_BASE_URL", DEFAULT_BASE_URL).strip()
    return raw if not raw else raw.rstrip("/") + "/"


def _ttl() -> float:
    try:
        return float(os.getenv("AQI_ARTIFACTS_TTL", "300"))
    except ValueError:
        return 300.0


def _timeout() -> float:
    try:
        return float(os.getenv("AQI_ARTIFACTS_TIMEOUT", "6"))
    except ValueError:
        return 6.0


# rel_path -> (fetched_at, payload, source)
_CACHE: dict[str, tuple[float, Any, str]] = {}
_LOCK = threading.Lock()


def _fetch(rel_path: str) -> Any:
    """Read one artefact from the repository. Raises on any failure."""
    import requests

    url = _base_url() + rel_path
    response = requests.get(url, timeout=_timeout())
    response.raise_for_status()
    return response.json()


def _read_local(local_path: Path) -> Any:
    return json.loads(local_path.read_text(encoding="utf-8"))


def load(rel_path: str, local_path: Path, *, default: Any = None) -> Any:
    """Return a published artefact, preferring the repository copy.

    Order of preference: a cached remote read that is still within its TTL, a
    fresh remote read, the copy bundled in the image, then ``default``. A
    remote failure is logged once per attempt and never raised, so a network
    problem degrades freshness rather than availability.
    """
    now = time.time()
    with _LOCK:
        cached = _CACHE.get(rel_path)
    if cached and now - cached[0] < _ttl():
        return cached[1]

    if _base_url():
        try:
            payload = _fetch(rel_path)
            with _LOCK:
                _CACHE[rel_path] = (now, payload, "repository")
            return payload
        except Exception as exc:  # noqa: BLE001 - never fail a request on this
            logger.warning("Could not read published %s (%s); using bundled copy.",
                           rel_path, exc)

    if local_path.exists():
        try:
            payload = _read_local(local_path)
            with _LOCK:
                _CACHE[rel_path] = (now, payload, "bundled")
            return payload
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bundled %s is unreadable (%s).", local_path.name, exc)

    # Nothing fresh and nothing bundled: hand back the stale cache if we have
    # one, so a transient outage does not turn into an empty dashboard.
    if cached:
        return cached[1]
    return default


def source_of(rel_path: str) -> str | None:
    """Where the currently cached copy of ``rel_path`` came from, if anywhere."""
    with _LOCK:
        cached = _CACHE.get(rel_path)
    return cached[2] if cached else None


def clear_cache() -> None:
    """Drop every cached artefact (used by tests and by manual refreshes)."""
    with _LOCK:
        _CACHE.clear()
