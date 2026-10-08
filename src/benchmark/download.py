"""Download the METR-LA benchmark inputs (idempotent).

Sources kept separate, as in the main pipeline:
  * METR-LA speeds       -> data/raw/metr_la/
  * Open-Meteo weather   -> data/raw/metr_weather/
  * US holidays (2012)   -> data/raw/metr_calendar/

Run: ``python -m src.benchmark.download``
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone

import holidays
import pandas as pd
import requests

from src.utils.config import project_path
from src.utils.log import get_logger

log = get_logger(__name__)

METR_URLS = [
    "https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic/metr_la.zip",
    "https://graphmining.ai/temporal_datasets/METR-LA.zip",
]
LA_LAT, LA_LON = 34.05, -118.25
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = "temperature_2m,precipitation,relative_humidity_2m"
TIMEZONE = "America/Los_Angeles"

METR_DIR = project_path("data/raw/metr_la")
WEATHER_DIR = project_path("data/raw/metr_weather")
CALENDAR_DIR = project_path("data/raw/metr_calendar")
H5_PATH = METR_DIR / "metr_la.h5"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def download_metr_la() -> None:
    """Fetch and unzip METR-LA, trying each mirror in turn."""
    if H5_PATH.exists():
        log.info("METR-LA already present, skipping download")
        return
    METR_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = METR_DIR / "metr_la.zip"
    for url in METR_URLS:
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(METR_DIR)
            log.info("downloaded METR-LA from %s", url.split("/")[2])
            return
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            log.warning("METR-LA mirror %s failed: %s", url.split("/")[2], type(exc).__name__)
    raise RuntimeError("both METR-LA mirrors failed")


def metr_date_range() -> tuple[str, str]:
    idx = pd.read_hdf(H5_PATH).index
    return idx.min().strftime("%Y-%m-%d"), idx.max().strftime("%Y-%m-%d")


def fetch_weather(start: str, end: str) -> None:
    """Save the raw Open-Meteo archive response, wrapped with ingestion metadata."""
    path = WEATHER_DIR / f"open_meteo_la_{start.replace('-', '')}_{end.replace('-', '')}.json"
    if path.exists():
        log.info("weather already present, skipping")
        return
    params = {
        "latitude": LA_LAT, "longitude": LA_LON, "start_date": start, "end_date": end,
        "hourly": WEATHER_VARS, "timezone": TIMEZONE,
    }
    resp = requests.get(WEATHER_URL, params=params, timeout=60)
    resp.raise_for_status()
    record = {
        "source": "open_meteo_archive", "schema_version": 1, "status": "ok",
        "ingested_at": _now_iso(), "request_params": params, "http_status": resp.status_code,
        "response": resp.json(),
    }
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    log.info("saved weather: %s", path.name)


def save_holidays(year: int = 2012) -> None:
    path = CALENDAR_DIR / f"us_holidays_{year}.csv"
    if path.exists():
        log.info("holidays already present, skipping")
        return
    rows = sorted(holidays.US(years=year).items())
    df = pd.DataFrame(rows, columns=["date", "name"])
    df["source"], df["schema_version"], df["ingested_at"] = "holidays_lib_US", 1, _now_iso()
    CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("saved %d holidays: %s", len(df), path.name)


def main() -> int:
    download_metr_la()
    start, end = metr_date_range()
    log.info("METR-LA range: %s to %s", start, end)
    fetch_weather(start, end)
    save_holidays(2012)
    return 0


if __name__ == "__main__":
    sys.exit(main())
