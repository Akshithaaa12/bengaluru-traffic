"""Fault-injection demo: two optional sources fail and the pipeline still completes, with flags and traces.

(a) the weather API fails for 3 days (retries exhausted -> quarantine record; those hours are null),
(b) the calendar file is missing. The pipeline runs on a small slice of METR-LA and
``reports/benchmark/fault_injection_demo.md`` records what failed, how it was detected, the flags set
and the outcome. Real raw/quarantine folders are never touched (everything happens in a temp dir).

Run: ``python -m src.benchmark.fault_demo``
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.benchmark import download as dl
from src.benchmark.dq_rules import weather_null_counts
from src.benchmark.prepare import build_full_table, load_raw_speeds
from src.utils.config import project_path
from src.utils.log import get_logger

log = get_logger(__name__)
OUT = project_path("reports/benchmark")
SLICE_DAYS = 14


class FailingSession:
    """Stands in for the Open-Meteo API during the outage: always HTTP 503."""

    def get(self, url: str, params: dict[str, Any], timeout: int) -> Any:  # noqa: ARG002
        return type("Resp", (), {"status_code": 503, "json": lambda self: {}})()


def inject_weather_outage(weather_dir: Path, tmp: Path, start: str, days: int) -> dict[str, Any]:
    """Quarantine a failed 3-day fetch and write a weather file whose hours in that window are null."""
    end = (pd.Timestamp(start) + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
    dl.fetch_weather(start, end, session=FailingSession(), sleep=lambda _: None,
                     out_dir=tmp / "unused_raw", quarantine_dir=tmp / "quarantine")
    quarantine = json.loads(next((tmp / "quarantine").glob("*.json")).read_text())

    rec = json.loads(next(weather_dir.glob("open_meteo_la_*.json")).read_text())
    hourly = rec["response"]["hourly"]
    times = pd.to_datetime(hourly["time"])
    window = (times >= pd.Timestamp(start)) & (times < pd.Timestamp(end) + pd.Timedelta(days=1))
    for var in hourly:
        if var != "time":
            hourly[var] = [None if w else v for v, w in zip(hourly[var], window)]
    out = tmp / "weather"
    out.mkdir()
    (out / "open_meteo_la_demo.json").write_text(json.dumps(rec), encoding="utf-8")
    return {"dir": out, "quarantine": quarantine, "start": start, "end": end, "hours_nulled": int(window.sum())}


def run_demo(
    raw: pd.DataFrame, weather_dir: Path, calendar_dir: Path, out_md: Path | None = None,
    outage_start: str = "2012-03-05", outage_days: int = 3,
) -> dict[str, Any]:
    """Run baseline and faulted pipelines on ``raw`` and describe the difference."""
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        base, base_stats = build_full_table(raw, weather_dir, calendar_dir)
        fault_w = inject_weather_outage(weather_dir, tmp, outage_start, outage_days)
        empty_calendar = tmp / "calendar_missing"
        empty_calendar.mkdir()
        faulted, stats = build_full_table(raw, fault_w["dir"], empty_calendar)   # must not raise

        hourly = json.loads(next(fault_w["dir"].glob("*.json")).read_text())["response"]["hourly"]
        n_sensors = faulted["sensor_id"].nunique()
        result = {
            "completed": True, "quarantine": fault_w["quarantine"], "hours_nulled": fault_w["hours_nulled"],
            "null_counts": weather_null_counts(hourly),
            "weather_unavailable_rows": int((faulted["weather_available"] == 0).sum()),
            "weather_unavailable_timestamps": int((faulted["weather_available"] == 0).sum() // n_sensors),
            "holiday_unavailable_rows": int((faulted["holiday_available"] == 0).sum()),
            "rows": len(faulted), "usable_base": int(base["usable"].sum()), "usable_fault": int(faulted["usable"].sum()),
            "speed_flags_unchanged": bool((base["sensor_missing"] == faulted["sensor_missing"]).all()
                                          and (base["interpolated"] == faulted["interpolated"]).all()),
            "base_weather_unavailable": int((base["weather_available"] == 0).sum()),
            "outage": f"{fault_w['start']} to {fault_w['end']}", "n_sensors": n_sensors, "days": len(raw) // 288,
        }
    if out_md is not None:
        write_markdown(result, out_md)
    return result


def write_markdown(r: dict[str, Any], path: Path) -> None:
    q = r["quarantine"]
    text = f"""# Fault-injection demo

Two **optional** sources are broken on purpose while the pipeline runs on a {r['days']}-day slice of METR-LA
({r['n_sensors']} sensors, {r['rows']:,} rows). Nothing in the real raw or quarantine folders is touched.

| | Fault | Source |
|---|---|---|
| (a) | Weather API failing (HTTP 503) for 3 days: {r['outage']} | `metr_weather` |
| (b) | Calendar file missing | `metr_calendar` |

## Which source failed and how it was detected

| Source | Detection | Evidence |
|---|---|---|
| metr_weather | Retry x3 with backoff exhausted, then quarantined (R4) | quarantine record: `error_type={q['error_type']}`, `http_status={q['http_status']}`, `attempts={q['attempts']}`, `status={q['status']}` |
| metr_weather | Null hourly values in the saved file (R5) | {r['hours_nulled']} hours null per variable ({', '.join(f"{v}: {n[1]}" for v, n in r['null_counts'].items())}) |
| metr_weather | No valid weather for the timestamp after the hour join (R6) | `weather_available=0` on {r['weather_unavailable_timestamps']:,} timestamps |
| metr_calendar | Calendar file not found (R10) | `holiday_available=0` on {r['holiday_unavailable_rows']:,} rows (all) |

## Flags set

| Flag | Rows | Meaning |
|---|---|---|
| `weather_available=0` | {r['weather_unavailable_rows']:,} (baseline: {r['base_weather_unavailable']:,}) | no valid weather; weather features NaN (forward-fill <= 1 h only) |
| `holiday_available=0` | {r['holiday_unavailable_rows']:,} | calendar unavailable; `is_holiday` defaulted to 0 (rare event, flagged, not silently filled) |
| `sensor_missing`, `interpolated` | unchanged: {r['speed_flags_unchanged']} | the speed source was unaffected |

## Outcome

- The pipeline **completed without raising**; no crash from the missing weather window or the missing calendar.
- Usable modelling rows: {r['usable_base']:,} without faults, {r['usable_fault']:,} with faults
  ({r['usable_base'] - r['usable_fault']:,} rows excluded because required weather features were NaN; rows are
  excluded, not guessed).
- The failure is traceable: the quarantine record names the source, error and attempts; the curated rows carry the flags.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    raw = load_raw_speeds().iloc[: SLICE_DAYS * 288]
    result = run_demo(raw, dl.WEATHER_DIR, dl.CALENDAR_DIR, OUT / "fault_injection_demo.md")
    log.info("demo complete: usable rows %d -> %d", result["usable_base"], result["usable_fault"])
    print((OUT / "fault_injection_demo.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
