"""Build the curated integrated dataset and the DQ outputs.

Writes ``data/curated/metr_curated.parquet`` (every row, with flags), the data dictionary
(``reports/benchmark/data_dictionary.csv``: source of every column), ``dq_rules.csv`` and
``missing_by_source.csv``.

Run: ``python -m src.benchmark.curate``
"""
from __future__ import annotations

import json
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.benchmark.download import QUARANTINE_DIR, WEATHER_DIR
from src.benchmark.dq_rules import RULES, missing_by_source, sensors_without_coordinates, timestamp_issues, weather_null_counts
from src.benchmark.prepare import build_full_table, load_raw_speeds
from src.utils.config import project_path
from src.utils.log import get_logger

log = get_logger(__name__)
OUT = project_path("reports/benchmark")
CURATED = project_path("data/curated/metr_curated.parquet")
LOCATIONS = project_path("data/raw/metr_la/sensor_locations_la.csv")

# (column, source, feature_group, description)
DATA_DICTIONARY: list[tuple[str, str, str, str]] = [
    ("sensor_id", "metr_la", "key", "Loop-detector id (integration key part 1)"),
    ("timestamp", "metr_la", "key", "5-min timestamp, naive local time (integration key part 2)"),
    ("speed_mph", "metr_la", "speed", "Valid speed; NaN where the sensor failed (speed 0)"),
    ("speed_filled_mph", "metr_la", "speed", "Speed after interpolating gaps <= 30 min; NaN for longer gaps"),
    ("sensor_missing", "metr_la", "dq_flag", "1 if the raw speed was 0/NaN (sensor failure)"),
    ("interpolated", "metr_la", "dq_flag", "1 if the value was filled by interpolation (gap <= 30 min)"),
    ("speed_out_of_range", "metr_la", "dq_flag", "1 if the raw speed was outside 0-100 mph (set NaN before cleaning)"),
    ("free_flow_mph", "metr_la", "speed", "Sensor's 95th-percentile valid speed"),
    ("ratio_t", "metr_la", "speed_lags", "speed / free_flow now"),
    ("ratio_lag5", "metr_la", "speed_lags", "ratio 5 min ago"),
    ("ratio_lag15", "metr_la", "speed_lags", "ratio 15 min ago"),
    ("ratio_lag30", "metr_la", "speed_lags", "ratio 30 min ago"),
    ("ratio_lag60", "metr_la", "speed_lags", "ratio 60 min ago"),
    ("ratio_mean_1h", "metr_la", "speed_rolling", "Rolling 1 h mean of ratio"),
    ("ratio_std_1h", "metr_la", "speed_rolling", "Rolling 1 h std of ratio"),
    ("hour_sin", "metr_la (timestamp)", "calendar_time", "sin of hour of day"),
    ("hour_cos", "metr_la (timestamp)", "calendar_time", "cos of hour of day"),
    ("wd_sin", "metr_la (timestamp)", "calendar_time", "sin of weekday"),
    ("wd_cos", "metr_la (timestamp)", "calendar_time", "cos of weekday"),
    ("is_peak", "metr_la (timestamp)", "calendar_time", "Weekday 07-10 or 16-20"),
    ("temp_c", "metr_weather", "weather", "Temperature 2 m (C), joined on the hour"),
    ("precip_mm", "metr_weather", "weather", "Precipitation (mm), joined on the hour"),
    ("humidity", "metr_weather", "weather", "Relative humidity 2 m (%), joined on the hour"),
    ("weather_available", "metr_weather", "dq_flag", "1 if a valid hourly weather row existed for this timestamp"),
    ("weather_out_of_range", "metr_weather", "dq_flag", "1 if a weather value was out of range (set NaN)"),
    ("holiday_available", "metr_calendar", "dq_flag", "1 if the calendar file was available (0 -> is_holiday defaulted to 0)"),
    ("is_holiday", "metr_calendar", "holiday", "1 if the date is a US holiday (joined on date); no match = valid 0"),
    ("y_now", "metr_la", "target", "Congestion class now (0 Low, 1 Moderate, 2 Severe)"),
    ("y", "metr_la", "target", "Congestion class 30 min ahead (target)"),
    ("usable", "integration", "dq_flag", "1 if all required features and the target are present (used for modelling)"),
]


def source_manifests() -> dict[str, dict[str, str | None]]:
    """Manifest path and ingested_at of each raw source, stored in the curated file's metadata (traceability)."""
    out: dict[str, dict[str, str | None]] = {}
    for name in ("metr_la", "metr_weather", "metr_calendar"):
        path = project_path("data/raw") / name / "_manifest.json"
        out[name] = {
            "manifest": str(path.relative_to(project_path("."))),
            "ingested_at": json.loads(path.read_text())["ingested_at"] if path.exists() else None,
        }
    return out


def count_quarantined_weather() -> int:
    return len(list(QUARANTINE_DIR.glob("*.json"))) if QUARANTINE_DIR.exists() else 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    full, stats = build_full_table()
    full = full.rename(columns={"time": "timestamp"})
    cols = [c for c, *_ in DATA_DICTIONARY]
    curated = full[cols].copy()
    for c in ("y", "y_now"):
        curated[c] = curated[c].astype("float32")

    source_map = {c: {"source": src, "group": grp} for c, src, grp, _ in DATA_DICTIONARY}
    table = pa.Table.from_pandas(curated, preserve_index=False)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        b"source_map": json.dumps(source_map).encode(),
        b"source_manifests": json.dumps(source_manifests()).encode(),
    })
    CURATED.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, CURATED, compression="zstd")

    pd.DataFrame(DATA_DICTIONARY, columns=["column", "source", "feature_group", "description"]).to_csv(
        OUT / "data_dictionary.csv", index=False
    )
    pd.DataFrame(RULES).to_csv(OUT / "dq_rules.csv", index=False)

    raw_all = load_raw_speeds()
    loc_ids = pd.read_csv(LOCATIONS, dtype={"sensor_id": str})["sensor_id"].tolist()
    no_coords = sensors_without_coordinates(raw_all.columns.astype(str).tolist(), loc_ids)
    hourly = json.loads(next(WEATHER_DIR.glob("open_meteo_la_*.json")).read_text())["response"]["hourly"]
    failed = count_quarantined_weather()
    missing = missing_by_source(
        stats, timestamp_issues(raw_all.index), len(no_coords), raw_all.shape[1],
        weather_null_counts(hourly), weather_requests_failed=failed, weather_requests=1 + failed,
    )
    missing.to_csv(OUT / "missing_by_source.csv", index=False)
    log.info("curated: %d rows x %d cols -> %s (%.1f MB)", len(curated), curated.shape[1], CURATED.name,
             CURATED.stat().st_size / 1e6)
    print(missing.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
