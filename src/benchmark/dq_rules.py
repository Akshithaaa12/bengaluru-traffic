"""Explicit data-quality rules for the benchmark sources.

Mandatory fields must be present for a record to be used at all; optional fields only enrich it
and are flagged when missing. Each rule states how a problem is detected and handled.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

MANDATORY: dict[str, list[str]] = {"metr_la": ["sensor_id", "timestamp", "speed"]}
OPTIONAL: dict[str, list[str]] = {
    "metr_weather": ["temperature_2m", "precipitation", "relative_humidity_2m"],
    "metr_calendar": ["holiday"],
}
STEP = pd.Timedelta(minutes=5)

# rule_id, source, field, requirement, detection, handling
RULES: list[dict[str, str]] = [
    {"rule_id": "R1", "source": "metr_la", "field": "speed", "requirement": "mandatory",
     "detection": "speed == 0 (or NaN): a loop detector reporting 0 mph is a sensor failure, never a real zero",
     "handling": "flag sensor_missing=1 and set NaN; interpolate gaps <= 30 min (interpolated=1); exclude longer gaps"},
    {"rule_id": "R2", "source": "metr_la", "field": "timestamp", "requirement": "mandatory",
     "detection": "null, duplicated or not on the regular 5-min grid", "handling": "reject the record (cannot be placed in time)"},
    {"rule_id": "R3", "source": "metr_la", "field": "sensor_id", "requirement": "mandatory",
     "detection": "missing, or no entry in sensor_locations", "handling": "exclude the sensor"},
    {"rule_id": "R4", "source": "metr_weather", "field": "api_request", "requirement": "optional",
     "detection": "timeout, connection error, HTTP 429 or 5xx",
     "handling": "retry x3 with exponential backoff, then quarantine; weather_available=0 for all rows"},
    {"rule_id": "R5", "source": "metr_weather", "field": "temperature_2m, precipitation, relative_humidity_2m",
     "requirement": "optional", "detection": "null hourly value",
     "handling": "weather_available=0; forward-fill <= 1 h only; rows still missing are excluded from modelling"},
    {"rule_id": "R6", "source": "metr_weather", "field": "join_on_hour", "requirement": "optional",
     "detection": "no weather row for floor(timestamp, 1 h)", "handling": "same as R5"},
    {"rule_id": "R7", "source": "metr_calendar", "field": "holiday", "requirement": "optional",
     "detection": "date absent from the holiday list", "handling": "valid is_holiday=0 (not missing)"},
    {"rule_id": "R8", "source": "integration", "field": "features + target", "requirement": "derived",
     "detection": "NaN in any required feature or the 30-min-ahead target (lags/rolling/target touch a long gap)",
     "handling": "exclude the row from train/val/test (kept in the curated table with usable=0)"},
]


def speed_failure_mask(raw: pd.DataFrame) -> pd.DataFrame:
    """R1: True where a speed reading is a sensor failure (0 or NaN)."""
    return raw.eq(0) | raw.isna()


def timestamp_issues(index: pd.DatetimeIndex) -> dict[str, int]:
    """R2: counts of null, duplicated and off-grid timestamps."""
    idx = pd.DatetimeIndex(index)
    steps = idx.to_series().diff().dropna()
    return {"total": len(idx), "null": int(idx.isna().sum()), "duplicated": int(idx.duplicated().sum()),
            "irregular_steps": int((steps != STEP).sum())}


def sensors_without_coordinates(sensor_ids: list[str], located_ids: list[str]) -> list[str]:
    """R3: sensors with no location record."""
    known = set(located_ids)
    return [s for s in sensor_ids if s not in known]


def weather_null_counts(hourly: dict[str, list[Any]]) -> dict[str, tuple[int, int]]:
    """R5: {variable: (hours, null hours)} from an Open-Meteo `hourly` block."""
    return {v: (len(vals), int(sum(x is None for x in vals))) for v, vals in hourly.items() if v != "time"}


def missing_by_source(
    stats: dict[str, Any], ts: dict[str, int], no_coords: int, n_sensors_total: int,
    weather_nulls: dict[str, tuple[int, int]], weather_requests_failed: int, weather_requests: int,
) -> pd.DataFrame:
    """One row per (source, check): how many records were missing/failed, how detected, how handled."""
    s = stats
    pct = lambda a, b: round(100 * a / b, 2) if b else 0.0  # noqa: E731
    rows = [
        ("metr_la", "speed == 0, all 207 sensors", "mandatory", s["raw_all_readings"], s["raw_all_zero"],
         "R1: speed == 0", "flag + set NaN", "-", "-"),
        ("metr_la", "speed == 0 / NaN, 40 selected sensors", "mandatory", s["readings"], s["sensor_missing"],
         "R1: speed == 0 or NaN", "flag sensor_missing=1; interpolate gaps <= 30 min; exclude longer",
         f"{s['interpolated']:,} interpolated", f"{s['still_nan']:,} excluded"),
        ("metr_la", "timestamp null / duplicated / irregular", "mandatory", ts["total"],
         ts["null"] + ts["duplicated"] + ts["irregular_steps"], "R2: isnull, duplicated, step != 5 min",
         "reject record", "-", "0 rejected" if not (ts["null"] + ts["duplicated"] + ts["irregular_steps"]) else "rejected"),
        ("metr_la", "sensor without coordinates", "mandatory", n_sensors_total, no_coords,
         "R3: no row in sensor_locations", "exclude sensor", "-", f"{no_coords} excluded"),
        ("metr_weather", "API request failed", "optional", weather_requests, weather_requests_failed,
         "R4: timeout / 429 / 5xx", "retry x3 then quarantine; weather_available=0", "-", "-"),
        *[("metr_weather", f"{var} null hours", "optional", total, nulls, "R5: null hourly value",
           "weather_available=0; ffill <= 1 h; else exclude", "-", "-")
          for var, (total, nulls) in weather_nulls.items()],
        ("metr_weather", "timestamps with no weather after the hour join", "optional", s["n_timestamps"],
         s["weather_unavailable_rows"], "R6: no row for floor(timestamp, 1 h)",
         "weather_available=0; ffill <= 1 h; else exclude", "-", "-"),
        ("metr_calendar", "date not a holiday", "optional", s["n_timestamps"], 0,
         "R7: absent date", "valid is_holiday=0 (not missing)", "-", "-"),
        ("integration", "rows with a NaN required feature or target", "derived", s["rows_total"],
         s["rows_total"] - s["rows_complete"], "R8: NaN in features/target",
         "exclude from modelling; keep in curated with usable=0", "-", f"{s['rows_total'] - s['rows_complete']:,} excluded"),
    ]
    df = pd.DataFrame(rows, columns=[
        "source", "check", "requirement", "total_records", "missing_or_failed", "detected_by", "handling",
        "filled", "excluded",
    ])
    df.insert(5, "pct", [pct(m, t) for m, t in zip(df["missing_or_failed"], df["total_records"])])
    return df
