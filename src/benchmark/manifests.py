"""Write raw-zone manifests, the sources table and the per-source schema report.

Outputs: ``data/raw/<source>/_manifest.json`` (metr_la, metr_weather, metr_calendar),
``reports/benchmark/sources.csv`` and ``reports/benchmark/source_schemas.md``.

Run: ``python -m src.benchmark.manifests``
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import holidays
import pandas as pd

from src.benchmark.curate import DATA_DICTIONARY
from src.benchmark.download import CALENDAR_DIR, H5_PATH, METR_DIR, METR_URLS, WEATHER_DIR, WEATHER_URL
from src.utils.config import project_path
from src.utils.log import get_logger

log = get_logger(__name__)
OUT = project_path("reports/benchmark")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _files(directory: Path) -> list[dict[str, Any]]:
    return [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(directory.iterdir())
            if p.is_file() and not p.name.startswith("_")]


def _csv_schema(path: Path, **kw: Any) -> dict[str, str]:
    return {c: str(t) for c, t in pd.read_csv(path, nrows=1000, **kw).dtypes.items()}


def metr_la_manifest() -> dict[str, Any]:
    speeds = pd.read_hdf(H5_PATH)
    loc, dist = METR_DIR / "sensor_locations_la.csv", METR_DIR / "distances_la.csv"
    return {
        "source": "metr_la", "format": "HDF5 (speeds, wide matrix) + CSV (locations, distances) + TXT",
        "ingestion_method": "file download (HTTP GET of a zip), unzipped unchanged", "url": METR_URLS[0],
        "mirror_url": METR_URLS[1], "ingested_at": _iso((METR_DIR / "metr_la.zip").stat().st_mtime),
        "files": _files(METR_DIR),
        "record_counts": {
            "metr_la.h5": {"rows": len(speeds), "columns": speeds.shape[1], "readings": int(speeds.size)},
            "sensor_locations_la.csv": {"rows": len(pd.read_csv(loc))},
            "distances_la.csv": {"rows": len(pd.read_csv(dist))},
        },
        "schema": {
            "metr_la.h5": {"index": f"timestamp ({speeds.index.dtype})", "columns": f"{speeds.shape[1]} sensor_id columns",
                           "value_dtype": str(speeds.dtypes.iloc[0]), "unit": "mph; 0 = sensor failure"},
            "sensor_locations_la.csv": _csv_schema(loc),
            "distances_la.csv": _csv_schema(dist),
        },
    }


def weather_manifest() -> dict[str, Any]:
    path = next(WEATHER_DIR.glob("open_meteo_la_*.json"))
    rec = json.loads(path.read_text())
    hourly = rec["response"]["hourly"]
    return {
        "source": "metr_weather", "format": "JSON (column-oriented hourly arrays)",
        "ingestion_method": "REST API (Open-Meteo archive), retry x3 + quarantine", "url": WEATHER_URL,
        "request_params": rec["request_params"], "ingested_at": rec["ingested_at"], "files": _files(WEATHER_DIR),
        "record_counts": {path.name: {"hourly_rows": len(hourly["time"])}},
        "schema": {path.name: {"time": "string (ISO local time)",
                               **{k: f"float ({rec['response']['hourly_units'].get(k, '')})" for k in hourly if k != "time"}}},
    }


def calendar_manifest() -> dict[str, Any]:
    path = next(CALENDAR_DIR.glob("us_holidays_*.csv"))
    df = pd.read_csv(path)
    return {
        "source": "metr_calendar", "format": "CSV", "ingestion_method": f"python library `holidays` {holidays.__version__} (US, 2012)",
        "url": "https://pypi.org/project/holidays/", "ingested_at": df["ingested_at"].iloc[0], "files": _files(CALENDAR_DIR),
        "record_counts": {path.name: {"rows": len(df)}}, "schema": {path.name: _csv_schema(path)},
    }


SOURCES = [
    # name, format, ingestion, raw zone, key, granularity, why useful, role
    ("Bengaluru TomTom live collector", "JSON (gzip)", "REST API polled every 15 min (GitHub Actions + cron-job.org)",
     "data/raw/traffic_api/", "(segment_key, timestamp_15min)", "15 min",
     "Primary source: live congestion for the real target city (13 South-East Bengaluru segments)",
     "Primary live source"),
    ("METR-LA sensors", "HDF5 matrix + CSV + TXT", "File download (zip)", "data/raw/metr_la/",
     "(sensor_id, timestamp_5min)", "5 min",
     "Benchmark training data: 4 months of loop-detector speeds to validate the model (no public Indian sensor history exists); "
     "locations/distances build the routing graph", "Benchmark training source"),
    ("Open-Meteo weather", "JSON (hourly arrays)", "REST API, retry x3 + quarantine", "data/raw/metr_weather/",
     "hour (timestamp floored to 1 h)", "1 h", "Rain/temperature/humidity can change speeds and congestion", "Benchmark feature source"),
    ("US holidays", "CSV", "`holidays` library", "data/raw/metr_calendar/",
     "date", "1 day", "Holiday traffic differs from normal weekdays", "Benchmark feature source"),
]

FEATURES_BY_SOURCE = {
    "metr_la": "speed_mph, speed_filled_mph, free_flow_mph, ratio_t, ratio_lag5/15/30/60, ratio_mean_1h, ratio_std_1h, "
               "sensor_missing, interpolated, y_now, y; timestamp-derived hour_sin/cos, wd_sin/cos, is_peak",
    "metr_weather": "temp_c, precip_mm, humidity, weather_available",
    "metr_calendar": "is_holiday",
}
KEYS = {"metr_la": "(sensor_id, timestamp_5min)", "metr_weather": "hour = floor(timestamp, 1 h)", "metr_calendar": "date = timestamp.date"}


def schema_table(schema: dict[str, Any]) -> str:
    rows = ["| Column | Type / unit |", "|---|---|"] + [f"| `{c}` | {t} |" for c, t in schema.items()]
    return "\n".join(rows)


def write_markdown(manifests: dict[str, dict[str, Any]]) -> None:
    out = ["# Source schemas and integration", "",
           "The sources differ in format, granularity and key; they are kept separate in raw zones and only joined in the "
           "curated table.", "",
           "| Source | Format | Granularity | Integration key |", "|---|---|---|---|"]
    out += [f"| {n} | {f} | {g} | {k} |" for n, f, _, _, k, g, _, _ in SOURCES]
    for name, m in manifests.items():
        out += ["", f"## {name}", "", f"Format: {m['format']}. Ingestion: {m['ingestion_method']}.", ""]
        for file, schema in m["schema"].items():
            out += [f"**`{file}`**", "", schema_table(schema), ""]
        out += [f"- **Integration key:** {KEYS[name]}", f"- **Features contributed:** {FEATURES_BY_SOURCE[name]}"]
    dd = pd.DataFrame(DATA_DICTIONARY, columns=["column", "source", "feature_group", "description"])
    out += ["", "## Curated dataset (`data/curated/metr_curated.parquet`) - source of every column", "",
            "| Column | Source | Group | Description |", "|---|---|---|---|"]
    out += [f"| `{r.column}` | {r.source} | {r.feature_group} | {r.description} |" for r in dd.itertuples()]
    (OUT / "source_schemas.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    manifests = {"metr_la": metr_la_manifest(), "metr_weather": weather_manifest(), "metr_calendar": calendar_manifest()}
    for name, m in manifests.items():
        path = project_path("data/raw") / name / "_manifest.json"
        path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        log.info("wrote %s", path.relative_to(project_path(".")))
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"METR-LA sensors": "34,272 x 207 readings", "Open-Meteo weather": "2,856 hourly rows",
              "US holidays": "12 dates", "Bengaluru TomTom live collector": "13 segments / run"}
    pd.DataFrame(
        [(n, r, f, i, z, k, g, w, counts[n]) for n, f, i, z, k, g, w, r in SOURCES],
        columns=["source", "role", "format", "ingestion", "raw_zone", "integration_key", "granularity", "why_useful", "records"],
    ).to_csv(OUT / "sources.csv", index=False)
    write_markdown(manifests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
