"""Collect one TomTom Flow Segment snapshot for every segment in settings.yaml.

One run writes one raw file (append-only, never overwritten):
``data/raw/traffic_api/YYYY-MM-DD/run_YYYYMMDDTHHMMZ.json``. Calls that still fail after retries
go to ``data/quarantine/traffic_api/run_YYYYMMDDTHHMMZ.json`` and the run carries on.

The API key and the full request URL are never logged or saved.

Run: ``python -m src.ingestion.tomtom_collector``
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from src.utils.config import get_env, get_settings, raw_dir, zone_dir
from src.utils.log import get_logger

log = get_logger(__name__)

TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
SOURCE = "tomtom_flow_v4"
SCHEMA_VERSION = 1
TIMEOUT_S = 10
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.0  # waits 1s, 2s, 4s between attempts


def _utc_iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def fetch_segment(
    session: requests.Session,
    segment: dict[str, Any],
    api_key: str,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Call the API for one segment with retries.

    Returns ``(record, None)`` on success or ``(None, quarantine_record)`` after the final failure.
    Retries (exponential backoff) on timeout, connection error, 429 and 5xx; other errors fail at once.
    Only the exception *type* is kept: requests' messages embed the URL, and with it the key.
    """
    key = segment["segment_key"]
    params = {"point": f"{segment['lat']},{segment['lon']}", "unit": "KMPH", "key": api_key}
    error_type, status = "unknown", None

    for attempt in range(MAX_RETRIES + 1):
        request_ts = _now()
        status = None
        try:
            resp = session.get(TOMTOM_URL, params=params, timeout=TIMEOUT_S)
            status = resp.status_code
            if status == 200:
                try:
                    body = resp.json()
                except ValueError:
                    error_type, retryable = "InvalidJSON", False
                else:
                    if isinstance(body, dict) and "flowSegmentData" in body:
                        return _success_record(key, request_ts, status, body), None
                    error_type, retryable = "MalformedResponse", False
            else:
                error_type, retryable = f"HTTP_{status}", _is_retryable_status(status)
        except (requests.Timeout, requests.ConnectionError) as exc:
            error_type, retryable = type(exc).__name__, True
        except requests.RequestException as exc:
            error_type, retryable = type(exc).__name__, False

        if not retryable or attempt == MAX_RETRIES:
            break
        delay = BACKOFF_BASE_S * 2**attempt
        log.warning("%s: %s (attempt %d/%d), retrying in %.0fs", key, error_type, attempt + 1, MAX_RETRIES + 1, delay)
        sleep(delay)

    log.error("%s: giving up after %d attempt(s): %s", key, attempt + 1, error_type)
    return None, {
        "segment_key": key,
        "request_ts_utc": _utc_iso(request_ts),
        "error_type": error_type,
        "http_status": status,
        "attempts": attempt + 1,
        "source": SOURCE,
        "schema_version": SCHEMA_VERSION,
        "ingested_at": _utc_iso(_now()),
    }


def _success_record(key: str, request_ts: datetime, status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_key": key,
        "request_ts_utc": _utc_iso(request_ts),
        "http_status": status,
        "response": body,
        "source": SOURCE,
        "schema_version": SCHEMA_VERSION,
        "ingested_at": _utc_iso(_now()),
    }


def _write_json(path: Path, records: list[dict[str, Any]]) -> None:
    """Write via a temp file so a crash never leaves a half-written raw file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_once(
    segments: list[dict[str, Any]],
    api_key: str,
    raw_root: Path,
    quarantine_root: Path,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    started: datetime | None = None,
) -> tuple[int, int, Path | None]:
    """Collect all segments and write the run file. Returns ``(n_ok, n_quarantined, raw_path)``.

    Idempotent: if this run's file already exists (same UTC minute) nothing is called or written.
    """
    started = started or _now()
    stamp = started.strftime("%Y%m%dT%H%MZ")
    raw_path = raw_root / started.strftime("%Y-%m-%d") / f"run_{stamp}.json"
    if raw_path.exists():
        log.warning("%s already exists; skipping this run", raw_path.name)
        return 0, 0, None

    session = session or requests.Session()
    records: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for seg in segments:
        record, failure = fetch_segment(session, seg, api_key, sleep)
        if record:
            records.append(record)
        else:
            quarantined.append(failure)

    if records:
        _write_json(raw_path, records)
    if quarantined:
        _write_json(quarantine_root / f"run_{stamp}.json", quarantined)
    return len(records), len(quarantined), raw_path if records else None


def format_summary(n_ok: int, n_quarantined: int) -> str:
    total = n_ok + n_quarantined
    label = "ok" if n_quarantined == 0 else ("FAILED" if n_ok == 0 else "partial")
    return f"run {label}: {n_ok}/{total} succeeded, {n_quarantined} quarantined"


def main() -> int:
    api_key = get_env("TOMTOM_API_KEY")
    if not api_key:
        log.error("TOMTOM_API_KEY is not set (put it in .env or GitHub Secrets)")
        return 1
    segments = get_settings().get("segments") or []
    if not segments:
        log.error("no segments in config/settings.yaml (run src.ingestion.select_segments)")
        return 1

    n_ok, n_quar, path = run_once(
        segments, api_key, raw_dir("traffic_api"), zone_dir("quarantine") / "traffic_api"
    )
    print(format_summary(n_ok, n_quar))
    if path:
        print(f"saved: {path}")
    return 0 if n_ok or not n_quar else 1


if __name__ == "__main__":
    sys.exit(main())
