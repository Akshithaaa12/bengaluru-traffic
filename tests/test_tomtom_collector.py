"""Tests for the TomTom collector with a mocked API (no network)."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from typing import Any

import requests

from src.ingestion import tomtom_collector as tc
from src.utils.raw_io import read_raw_run

API_KEY = "SECRET-KEY-123"
SEGMENTS = [
    {"segment_key": "seg_a", "lat": 12.91, "lon": 77.62},
    {"segment_key": "seg_b", "lat": 12.93, "lon": 77.61},
]
BODY = {"flowSegmentData": {"frc": "FRC2", "currentSpeed": 30, "freeFlowSpeed": 40, "confidence": 1}}
STARTED = datetime(2026, 10, 7, 12, 30, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, status: int = 200, body: Any = None, bad_json: bool = False):
        self.status_code, self._body, self._bad_json = status, body, bad_json

    def json(self) -> Any:
        if self._bad_json:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Serves scripted outcomes per segment (matched via the `point` param); items may be exceptions."""

    def __init__(self, script: dict[str, list[Any]]):
        self.script, self.calls = script, []

    def get(self, url: str, params: dict[str, str], timeout: int) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        lat = params["point"].split(",")[0]
        outcome = self.script[lat].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok() -> FakeResponse:
    return FakeResponse(200, BODY)


def run(tmp_path, script, sleeps=None):
    sleeps = [] if sleeps is None else sleeps
    session = FakeSession(script)
    result = tc.run_once(
        SEGMENTS, API_KEY, tmp_path / "raw", tmp_path / "quarantine",
        session=session, sleep=sleeps.append, started=STARTED,
    )
    return result, session, sleeps


def test_success_writes_one_file_with_all_fields(tmp_path):
    (n_ok, n_quar, path), session, sleeps = run(tmp_path, {"12.91": [ok()], "12.93": [ok()]})

    assert (n_ok, n_quar) == (2, 0)
    assert path == tmp_path / "raw" / "2026-10-07" / "run_20261007T1230Z.json.gz"
    records = read_raw_run(path)
    assert [r["segment_key"] for r in records] == ["seg_a", "seg_b"]
    first = records[0]
    assert set(first) == {
        "segment_key", "request_ts_utc", "http_status", "response", "source", "schema_version", "ingested_at",
    }
    assert first["http_status"] == 200
    assert first["response"] == BODY  # unmodified
    assert first["source"] == "tomtom_flow_v4" and first["schema_version"] == 1
    assert first["request_ts_utc"].endswith("Z") and first["ingested_at"].endswith("Z")
    assert not (tmp_path / "quarantine").exists()
    assert sleeps == []
    call = session.calls[0]
    assert call["timeout"] == 10 and call["params"]["unit"] == "KMPH" and call["params"]["point"] == "12.91,77.62"


def test_file_is_compact_gzipped_json_with_unmodified_response(tmp_path):
    (_, _, path), _, _ = run(tmp_path, {"12.91": [ok()], "12.93": [ok()]})

    raw = gzip.decompress(path.read_bytes()).decode()
    assert "\n" not in raw and '": ' not in raw and ", " not in raw  # compact separators, no indent
    assert raw == json.dumps(json.loads(raw), separators=(",", ":"))
    assert json.loads(raw)[0]["response"] == BODY


def test_gzip_output_is_deterministic(tmp_path):
    records = [{"a": 1}]
    tc._write_json_gz(tmp_path / "x.json.gz", records)
    tc._write_json_gz(tmp_path / "y.json.gz", records)
    assert (tmp_path / "x.json.gz").read_bytes() == (tmp_path / "y.json.gz").read_bytes()


def test_old_plain_json_run_in_same_minute_blocks_rerun(tmp_path):
    old = tmp_path / "raw" / "2026-10-07" / "run_20261007T1230Z.json"
    old.parent.mkdir(parents=True)
    old.write_text("[]")
    (n_ok, n_quar, path), session, _ = run(tmp_path, {"12.91": [], "12.93": []})
    assert (n_ok, n_quar, path) == (0, 0, None) and session.calls == []
    assert old.read_text() == "[]"


def test_429_then_success_retries_with_backoff(tmp_path):
    (n_ok, n_quar, path), session, sleeps = run(
        tmp_path, {"12.91": [FakeResponse(429), ok()], "12.93": [ok()]}
    )

    assert (n_ok, n_quar) == (2, 0)
    assert sleeps == [1.0]
    assert len(session.calls) == 3
    assert read_raw_run(path)[0]["http_status"] == 200


def test_timeout_then_success_is_retried(tmp_path):
    (n_ok, n_quar, _), _, sleeps = run(
        tmp_path, {"12.91": [requests.Timeout("boom"), ok()], "12.93": [ok()]}
    )
    assert (n_ok, n_quar) == (2, 0) and sleeps == [1.0]


def test_permanent_failure_is_quarantined_and_run_continues(tmp_path):
    script = {"12.91": [FakeResponse(503)] * 4, "12.93": [ok()]}
    (n_ok, n_quar, path), session, sleeps = run(tmp_path, script)

    assert (n_ok, n_quar) == (1, 1)
    assert sleeps == [1.0, 2.0, 4.0]  # 3 retries, exponential backoff
    assert len(session.calls) == 5  # 4 attempts for seg_a + 1 for seg_b
    assert [r["segment_key"] for r in read_raw_run(path)] == ["seg_b"]

    q = read_raw_run(tmp_path / "quarantine" / "run_20261007T1230Z.json.gz")
    assert len(q) == 1
    assert q[0]["segment_key"] == "seg_a"
    assert q[0]["error_type"] == "HTTP_503" and q[0]["http_status"] == 503 and q[0]["attempts"] == 4
    assert "response" not in q[0]


def test_non_retryable_error_fails_immediately(tmp_path):
    (n_ok, n_quar, _), session, sleeps = run(tmp_path, {"12.91": [FakeResponse(403)], "12.93": [ok()]})
    assert (n_ok, n_quar) == (1, 1) and sleeps == [] and len(session.calls) == 2


def test_bad_json_and_malformed_body_are_quarantined(tmp_path):
    script = {"12.91": [FakeResponse(200, bad_json=True)], "12.93": [FakeResponse(200, {"oops": 1})]}
    (n_ok, n_quar, path), _, _ = run(tmp_path, script)

    assert (n_ok, n_quar, path) == (0, 2, None)
    q = read_raw_run(tmp_path / "quarantine" / "run_20261007T1230Z.json.gz")
    assert [r["error_type"] for r in q] == ["InvalidJSON", "MalformedResponse"]


def test_timeout_after_all_retries_records_exception_type_without_url(tmp_path):
    leaky = requests.Timeout(f"HTTPSConnectionPool: {tc.TOMTOM_URL}?key={API_KEY}")
    script = {"12.91": [leaky] * 4, "12.93": [ok()]}
    (_, n_quar, _), _, _ = run(tmp_path, script)

    assert n_quar == 1
    q = read_raw_run(tmp_path / "quarantine" / "run_20261007T1230Z.json.gz")
    assert q[0]["error_type"] == "Timeout" and q[0]["http_status"] is None


def test_api_key_and_url_never_saved(tmp_path):
    run(tmp_path, {"12.91": [FakeResponse(500)] * 4, "12.93": [ok()]})
    files = list(tmp_path.rglob("*.json.gz"))
    assert files
    for f in files:
        text = gzip.decompress(f.read_bytes()).decode()
        assert API_KEY not in text and "api.tomtom.com" not in text


def test_existing_run_file_is_not_overwritten(tmp_path):
    run(tmp_path, {"12.91": [ok()], "12.93": [ok()]})
    (n_ok, n_quar, path), session, _ = run(tmp_path, {"12.91": [], "12.93": []})
    assert (n_ok, n_quar, path) == (0, 0, None) and session.calls == []


def test_format_summary():
    assert tc.format_summary(13, 0) == "run ok: 13/13 succeeded, 0 quarantined"
    assert tc.format_summary(11, 2) == "run partial: 11/13 succeeded, 2 quarantined"
    assert tc.format_summary(0, 13) == "run FAILED: 0/13 succeeded, 13 quarantined"
