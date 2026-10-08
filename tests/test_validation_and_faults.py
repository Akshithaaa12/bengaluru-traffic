"""Tests for range validation, graceful handling of missing optional sources, and the fault-injection demo."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.benchmark import dq_rules as dq
from src.benchmark import fault_demo
from src.benchmark.prepare import build_full_table, load_holiday_dates, load_weather, time_features

N_SENSORS = 40


def make_raw(days: int = 6) -> pd.DataFrame:
    idx = pd.date_range("2012-03-01", periods=days * 288, freq="5min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.uniform(30, 65, (len(idx), N_SENSORS)).round(1), index=idx,
                        columns=[str(i) for i in range(N_SENSORS)])


def write_weather(directory, days: int = 6, **overrides) -> None:
    times = pd.date_range("2012-03-01", periods=days * 24, freq="h")
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times], "temperature_2m": [15.0] * len(times),
              "precipitation": [0.0] * len(times), "relative_humidity_2m": [60.0] * len(times)}
    for k, v in overrides.items():
        hourly[k][0] = v
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "open_meteo_la_synthetic.json").write_text(json.dumps({"response": {"hourly": hourly}}))


def write_calendar(directory) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2012-03-03"], "name": ["x"]}).to_csv(directory / "us_holidays_2012.csv", index=False)


# --- R9 validation -----------------------------------------------------------------------

def test_speed_out_of_range_becomes_nan_and_flagged_zero_is_left_to_r1():
    raw = pd.DataFrame({"a": [150.0, -5.0, 0.0, 55.0]}, index=pd.date_range("2012-03-01", periods=4, freq="5min"))
    out = dq.validate_speeds(raw)
    assert out["speed_out_of_range"]["a"].tolist() == [True, True, False, False]
    assert out["speeds"]["a"].isna().tolist() == [True, True, False, False]
    assert out["speeds"]["a"].iloc[2] == 0.0  # zero is a sensor failure (R1), not an out-of-range value


def test_duplicate_and_unsorted_timestamps_are_cleaned_and_counted():
    idx = pd.DatetimeIndex(["2012-03-01 00:05", "2012-03-01 00:00", "2012-03-01 00:00"])
    out = dq.validate_speeds(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx))
    assert out["index_counts"] == {"non_monotonic": 2,  # steps that are not strictly increasing (the drop and the repeat)
                                     "duplicate_timestamps": 1, "duplicate_sensor_columns": 0}
    assert out["speeds"].index.is_monotonic_increasing and out["speeds"].index.is_unique
    assert out["speeds"]["a"].iloc[0] == 2.0  # first duplicate kept


def test_duplicate_sensor_columns_dropped():
    raw = pd.DataFrame([[1.0, 2.0]], columns=["s1", "s1"], index=pd.DatetimeIndex(["2012-03-01"]))
    out = dq.validate_speeds(raw)
    assert out["index_counts"]["duplicate_sensor_columns"] == 1 and list(out["speeds"].columns) == ["s1"]


def test_weather_ranges_humidity_temp_precip():
    wx = pd.DataFrame({"temperature_2m": [10.0, 80.0, -30.0], "relative_humidity_2m": [50.0, 120.0, 0.0],
                       "precipitation": [0.0, -1.0, 2.0]})
    cleaned, flag, counts = dq.validate_weather(wx)
    assert counts == {"temperature_2m": 2, "relative_humidity_2m": 1, "precipitation": 1}
    assert flag.tolist() == [0, 1, 1]
    assert cleaned.isna().sum().sum() == 4


def test_validate_ranges_combines_speed_and_weather():
    raw = make_raw(1).iloc[:3]
    raw.iloc[0, 0] = 999.0
    wx = pd.DataFrame({"relative_humidity_2m": [101.0]})
    out = dq.validate_ranges(raw, wx)
    assert out["speed_out_of_range"].iloc[0, 0] and out["weather_out_of_range_counts"] == {"relative_humidity_2m": 1}


def test_out_of_range_weather_flows_into_flags(tmp_path):
    write_weather(tmp_path, relative_humidity_2m=150.0)
    idx = pd.date_range("2012-03-01", periods=24, freq="5min")  # first two hours
    out = load_weather(idx, tmp_path)
    assert out["weather_out_of_range"].iloc[0] == 1 and out["weather_available"].iloc[0] == 0
    assert out["weather_available"].iloc[-1] == 1 and out.attrs["out_of_range_counts"]["relative_humidity_2m"] == 1


# --- R4 / R10: missing optional sources do not crash ---------------------------------------

def test_missing_weather_file_returns_nan_and_flag_zero(tmp_path):
    idx = pd.date_range("2012-03-01", periods=12, freq="5min")
    out = load_weather(idx, tmp_path)  # empty dir
    assert out[["temperature_2m", "precipitation", "relative_humidity_2m"]].isna().all().all()
    assert (out["weather_available"] == 0).all() and len(out) == 12


def test_missing_calendar_gives_flagged_default(tmp_path):
    assert load_holiday_dates(tmp_path) is None
    tf = time_features(pd.date_range("2012-03-03", periods=3, freq="5min"), None)
    assert (tf["holiday_available"] == 0).all() and (tf["is_holiday"] == 0).all()
    write_calendar(tmp_path)
    tf = time_features(pd.date_range("2012-03-03", periods=3, freq="5min"), load_holiday_dates(tmp_path))
    assert (tf["holiday_available"] == 1).all() and (tf["is_holiday"] == 1).all()


def test_pipeline_completes_with_both_optional_sources_missing(tmp_path):
    df, stats = build_full_table(make_raw(), tmp_path / "no_weather", tmp_path / "no_calendar")
    assert (df["weather_available"] == 0).all() and (df["holiday_available"] == 0).all()
    assert df["usable"].sum() == 0 and stats["holiday_unavailable_rows"] == len(df) // N_SENSORS


# --- fault-injection demo ------------------------------------------------------------------

def test_fault_demo_runs_and_reports(tmp_path):
    raw = make_raw()
    write_weather(tmp_path / "w")
    write_calendar(tmp_path / "c")
    out_md = tmp_path / "demo.md"
    r = fault_demo.run_demo(raw, tmp_path / "w", tmp_path / "c", out_md, outage_start="2012-03-02", outage_days=3)

    assert r["completed"] and out_md.exists()
    assert r["quarantine"]["error_type"] == "HTTP_503" and r["quarantine"]["attempts"] == 4
    assert r["weather_unavailable_timestamps"] == 3 * 288
    assert r["holiday_unavailable_rows"] == r["rows"]
    assert r["speed_flags_unchanged"] and r["usable_fault"] < r["usable_base"]
    md = out_md.read_text()
    assert "weather_available=0" in md and "holiday_available=0" in md and "completed without raising" in md
