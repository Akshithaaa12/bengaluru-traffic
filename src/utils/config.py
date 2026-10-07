"""Project configuration: settings.yaml, .env and project-root-relative paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
SETTINGS_PATH: Path = PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def get_settings() -> dict[str, Any]:
    """Load .env into the environment and return the parsed settings.yaml."""
    load_dotenv(PROJECT_ROOT / ".env")
    with SETTINGS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(*parts: str | os.PathLike[str]) -> Path:
    """Return a path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def raw_dir(source: str) -> Path:
    """Raw-zone directory for a source (traffic_api, weather, road_network, btp, calendar)."""
    return project_path(get_settings()["paths"]["raw"][source])


def zone_dir(name: str) -> Path:
    """Directory for a top-level zone: staging, curated, quarantine or reports."""
    return project_path(get_settings()["paths"][name])


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable (e.g. TOMTOM_API_KEY) after .env has been loaded."""
    get_settings()
    return os.environ.get(name, default)
