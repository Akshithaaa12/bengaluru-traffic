"""Read helpers for raw run files (plain ``.json`` from early runs, ``.json.gz`` since)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


def read_raw_run(path: str | Path) -> list[dict[str, Any]]:
    """Return the list of records in a raw run (or quarantine) file, plain or gzipped."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)
