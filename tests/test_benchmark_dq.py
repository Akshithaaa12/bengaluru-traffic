"""Tests for the benchmark DQ rules and the weather fetch (retry + quarantine)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import requests

from src.benchmark import download as dl
from src.benchmark import dq_rules as dq
from src.benchmark.prepare import fill_short_gaps


# --- DQ rules ----------------------------------------------------------------------------

def test_speed_zero_and_nan_are_failures_positive_is_not():
    raw = pd.DataFrame({"a": [0.0, 55.0, np.nan, 3.0]})
    assert dq.speed_failure_mask(raw)["a"].tolist() == [True, False, True, False]


def test_timestamp_issues_counts_duplicates_and_gaps():
    idx = pd.DatetimeIndex(["2012-03-01 00:00", "2012-03-01 00:05", "2012-03-01 00:05", "2012-03-01 00:20"])
    out = dq.timestamp_issues(idx)
    assert out["duplicated"] == 1 and out["null"] == 0
    assert out["irregular_steps"] == 2  # 0-step (duplicate) and 15-min jump


def test_timestamp_issues_clean_grid_is_zero():
    idx = pd.date_range("2012-03-01", periods=5, freq="5min")
    out = dq.timestamp_issues(idx)
    assert (out["null"], out["duplicated"], out["irregular_steps"]) == (0, 0, 0)


def test_sensors_without_coordinates():
    assert dq.sensors_without_coordinates(["1", "2", "3"], ["1", "3"]) == ["2"]


def test_weather_null_counts_ignores_time():
    hourly = {"time": ["t1", "t2", "t3"], "temperature_2m": [10.0, None, 12.0], "precipitation": [0.0, 0.0, 0.0]}
    assert dq.weather_null_counts(hourly) == {"temperature_2m": (3, 1), "precipitation": (3, 0)}


def test_gap_up_to_30_min_is_interpolated_longer_stays_nan():
    def series(gap: int) -> pd.Series:
        return pd.Series([10.0] + [np.nan] * gap + [20.0, 20.0])

    assert fill_short_gaps(series(6)).notna().all()          # 6 x 5 min = 30 min -> filled
    assert fill_short_gaps(series(7)).isna().sum() == 7      # 35 min -> left NaN (excluded)


def test_edge_gap_is_not_filled():
    s = pd.Series([np.nan, np.nan, 10.0, 10.0])
    assert fill_short_gaps(s).isna().sum() == 2


def test_missing_by_source_has_one_row_per_check_and_pct():
    stats = {"raw_all_readings": 100, "raw_all_zero": 8, "readings": 40, "sensor_missing": 4, "interpolated": 1,
             "still_nan": 3, "n_timestamps": 10, "weather_unavailable_rows": 0, "rows_total": 40, "rows_complete": 30}
    ts = {"total": 10, "null": 0, "duplicated": 0, "irregular_steps": 0}
    df = dq.missing_by_source(stats, ts, 0, 207, {"temperature_2m": (24, 0)}, 0, 1)
    assert set(df["source"]) == {"metr_la", "metr_weather", "metr_calendar", "integration"}
    assert df.loc[df["check"].str.startswith("speed == 0, all"), "pct"].iloc[0] == 8.0
    assert df.loc[df["source"] == "integration", "missing_or_failed"].iloc[0] == 10


# --- weather fetch: retry + quarantine ---------------------------------------------------

class FakeResp:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self._body = status, body or {}

    def json(self) -> dict:
        return self._body


class FakeSession:
    def __init__(self, outcomes: list):
        self.outcomes, self.calls = list(outcomes), 0

    def get(self, url, params, timeout):  # noqa: ANN001
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def fetch(tmp_path, outcomes):
    sleeps: list[float] = []
    session = FakeSession(outcomes)
    path = dl.fetch_weather("2012-03-01", "2012-03-02", session=session, sleep=sleeps.append,
                            out_dir=tmp_path / "raw", quarantine_dir=tmp_path / "q")
    return path, session, sleeps


def test_weather_retries_429_then_saves_with_metadata(tmp_path):
    path, session, sleeps = fetch(tmp_path, [FakeResp(429), FakeResp(200, {"hourly": {"time": []}})])
    rec = json.loads(path.read_text())
    assert session.calls == 2 and sleeps == [1.0]
    assert rec["source"] == "open_meteo_archive" and rec["http_status"] == 200 and "ingested_at" in rec
    assert not (tmp_path / "q").exists()


def test_weather_permanent_failure_is_quarantined_not_raised(tmp_path):
    path, session, sleeps = fetch(tmp_path, [requests.Timeout("secret-url")] * 4)
    assert path is None and session.calls == 4 and sleeps == [1.0, 2.0, 4.0]
    assert not (tmp_path / "raw").exists()
    (q,) = list((tmp_path / "q").glob("*.json"))
    rec = json.loads(q.read_text())
    assert rec["error_type"] == "Timeout" and rec["attempts"] == 4 and rec["status"] == "quarantined"
    assert "secret-url" not in q.read_text()


def test_weather_400_is_not_retried(tmp_path):
    path, session, sleeps = fetch(tmp_path, [FakeResp(400)])
    assert path is None and session.calls == 1 and sleeps == []
    (q,) = list((tmp_path / "q").glob("*.json"))
    assert json.loads(q.read_text())["http_status"] == 400
