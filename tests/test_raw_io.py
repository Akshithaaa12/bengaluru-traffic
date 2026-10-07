"""Tests for reading raw run files."""
from __future__ import annotations

import gzip
import json

from src.utils.raw_io import read_raw_run

RECORDS = [{"segment_key": "a", "response": {"x": [1, 2]}}]


def test_reads_plain_json(tmp_path):
    p = tmp_path / "run.json"
    p.write_text(json.dumps(RECORDS, indent=2))
    assert read_raw_run(p) == RECORDS


def test_reads_gzipped_json_and_accepts_str_path(tmp_path):
    p = tmp_path / "run.json.gz"
    p.write_bytes(gzip.compress(json.dumps(RECORDS, separators=(",", ":")).encode()))
    assert read_raw_run(str(p)) == RECORDS
