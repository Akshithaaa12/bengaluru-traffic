"""Select and validate the monitored road segments.

Geocodes the candidate locations (Nominatim), snaps each to the nearest major OSM edge
(trunk/primary/secondary), validates the snapped point with one TomTom Flow Segment call,
and writes the successful points to ``config/settings.yaml`` under ``segments``.

Run: ``python -m src.ingestion.select_segments``
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
import yaml
from shapely.geometry import Point
from shapely.ops import nearest_points

from src.utils.config import SETTINGS_PATH, get_env, get_settings, zone_dir
from src.utils.log import get_logger

log = get_logger(__name__)

CITY_SUFFIX = "Bengaluru, Karnataka, India"
REFERENCE_KEY = "silk_board"
MAX_DIST_KM = 15.0
MAX_SNAP_M = 50.0
HIGHWAY_TYPES = ("trunk", "primary", "secondary")
OVERPASS_FILTER = '["highway"~"^(trunk|primary|secondary)$"]'
NOMINATIM_DELAY_S = 1.0
TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
USER_AGENT = "bengaluru-traffic-ml2-project"

# (segment_key, name as geocoded)
LOCATIONS: list[tuple[str, str]] = [
    ("silk_board", "Silk Board Junction"),
    ("hsr_27th_main", "HSR Layout 27th Main Road"),
    ("agara_orr", "Agara Junction Outer Ring Road"),
    ("bellandur_orr", "Bellandur Outer Ring Road"),
    ("marathahalli_bridge", "Marathahalli Bridge"),
    ("kadubeesanahalli", "Kadubeesanahalli"),
    ("wipro_junction", "Wipro Junction Sarjapur Road"),
    ("iblur_junction", "Iblur Junction"),
    ("sony_world", "Sony World Junction Koramangala"),
    ("forum_mall", "Forum Mall Koramangala"),
    ("madiwala", "Madiwala Checkpost"),
    ("bommanahalli", "Bommanahalli Hosur Road"),
    ("ec_flyover", "Electronic City Flyover"),
    ("kudlu_gate", "Kudlu Gate"),
    ("haralur_road", "Haralur Road"),
    ("domlur_flyover", "Domlur Flyover"),
    ("hal_old_airport_rd", "HAL Old Airport Road"),
    ("ejipura_signal", "Ejipura Signal"),
    ("st_johns", "St John's Hospital Junction Koramangala"),
    ("udupi_garden", "Udupi Garden BTM Layout"),
]


@dataclass
class Result:
    """Outcome for one candidate location."""

    key: str
    name: str
    status: str = "OK"
    lat: float | None = None
    lon: float | None = None
    osm_road_name: str | None = None
    highway: str | None = None
    snap_m: float | None = None
    frc: str | None = None
    current_speed: float | None = None
    free_flow_speed: float | None = None
    confidence: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "OK"


# --- geocoding -------------------------------------------------------------------------

def _load_cache(path: Path) -> dict[str, list[float]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def geocode(name: str, cache: dict[str, list[float]], cache_path: Path) -> tuple[float, float] | None:
    """Geocode ``name`` via Nominatim (cached, 1 request/sec). Failures are not cached."""
    query = f"{name}, {CITY_SUFFIX}"
    if query in cache:
        return cache[query][0], cache[query][1]
    try:
        lat, lon = ox.geocode(query)
    except Exception as exc:  # osmnx raises InsufficientResponseError or requests errors
        log.warning("geocode failed for %r: %s", query, type(exc).__name__)
        return None
    finally:
        time.sleep(NOMINATIM_DELAY_S)
    cache[query] = [lat, lon]
    _save_cache(cache_path, cache)
    return lat, lon


# --- snapping --------------------------------------------------------------------------

def _first(value: Any) -> Any:
    """OSM tags may be lists after graph simplification; take the first element."""
    if isinstance(value, list):
        return value[0] if value else None
    return None if pd.isna(value) else value


def _pick_highway(value: Any) -> str | None:
    values = value if isinstance(value, list) else [value]
    return next((v for v in values if v in HIGHWAY_TYPES), None)


def load_major_edges(center: tuple[float, float], radius_m: float) -> gpd.GeoDataFrame:
    """Download trunk/primary/secondary edges around ``center`` and project to a metric CRS."""
    graph = ox.graph_from_point(
        center, dist=radius_m, custom_filter=OVERPASS_FILTER, retain_all=True, simplify=True
    )
    edges = ox.graph_to_gdfs(graph, nodes=False)
    edges["hw"] = edges["highway"].map(_pick_highway)
    edges = edges[edges["hw"].notna()]
    return ox.projection.project_gdf(edges)


def snap_to_edge(lat: float, lon: float, edges: gpd.GeoDataFrame) -> dict[str, Any] | None:
    """Return the nearest point on the nearest edge plus the edge's name/highway."""
    if edges.empty:
        return None
    point = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(edges.crs).iloc[0]
    pos = int(edges.geometry.distance(point).to_numpy().argmin())
    edge = edges.iloc[pos]
    snapped = nearest_points(edge.geometry, point)[0]
    snapped_ll = gpd.GeoSeries([snapped], crs=edges.crs).to_crs(4326).iloc[0]
    return {
        "lat": round(snapped_ll.y, 6),
        "lon": round(snapped_ll.x, 6),
        "osm_road_name": _first(edge.get("name")),
        "highway": edge["hw"],
        "snap_m": round(float(snapped.distance(point)), 1),
    }


# --- TomTom validation -----------------------------------------------------------------

def tomtom_flow(lat: float, lon: float, api_key: str) -> dict[str, Any]:
    """One Flow Segment call. Returns the fields we record, or ``{"error": ...}``.

    Exception text is never propagated because requests embeds the URL (and key) in it.
    """
    try:
        resp = requests.get(TOMTOM_URL, params={"key": api_key, "point": f"{lat},{lon}"}, timeout=20)
        if resp.status_code != 200:
            return {"error": f"TomTom HTTP {resp.status_code}"}
        seg = resp.json()["flowSegmentData"]
        return {
            "frc": seg.get("frc"),
            "current_speed": seg.get("currentSpeed"),
            "free_flow_speed": seg.get("freeFlowSpeed"),
            "confidence": seg.get("confidence"),
        }
    except (requests.RequestException, ValueError, KeyError) as exc:
        return {"error": f"TomTom {type(exc).__name__}"}


# --- outputs ---------------------------------------------------------------------------

def _segment_entry(r: Result) -> dict[str, Any]:
    return {
        "segment_key": r.key, "name": r.name, "lat": r.lat, "lon": r.lon,
        "osm_road_name": r.osm_road_name, "highway": r.highway,
    }


def write_segments(path: Path, new: list[dict[str, Any]]) -> None:
    """Replace the ``segments:`` block in settings.yaml, leaving every other line untouched.

    Existing entries whose key is not in ``new`` are kept, so manual additions survive a rerun.
    """
    text = path.read_text(encoding="utf-8")
    existing = (yaml.safe_load(text) or {}).get("segments") or []
    merged = {s["segment_key"]: s for s in existing}
    merged.update({s["segment_key"]: s for s in new})
    order = {k: i for i, (k, _) in enumerate(LOCATIONS)}
    entries = sorted(merged.values(), key=lambda s: order.get(s["segment_key"], len(order)))

    if entries:
        body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True)
        block = "segments:\n" + "".join(f"  {line}\n" for line in body.splitlines())
    else:
        block = "segments: []\n"
    pattern = re.compile(r"^segments:.*\n(?:[ \t]+.*\n)*", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError("no top-level 'segments:' key found in settings.yaml")
    new_text = pattern.sub(lambda _: block, text, count=1)

    before, after = yaml.safe_load(text), yaml.safe_load(new_text)
    before.pop("segments", None)
    after.pop("segments", None)
    if before != after:
        raise RuntimeError("refusing to write: other settings sections would change")
    path.write_text(new_text, encoding="utf-8")


def build_map(results: list[Result], out: Path) -> None:
    """Folium map with each successful point labelled."""
    ok = [r for r in results if r.ok]
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap = folium.Map(
        location=[sum(r.lat for r in ok) / len(ok), sum(r.lon for r in ok) / len(ok)],
        zoom_start=12, tiles="OpenStreetMap",
    )
    for r in ok:
        popup = f"{r.name}<br>{r.osm_road_name} ({r.highway})<br>{r.frc}, {r.current_speed}/{r.free_flow_speed} km/h"
        folium.CircleMarker(
            [r.lat, r.lon], radius=6, color="#c0392b", fill=True, fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=300),
            tooltip=folium.Tooltip(r.key, permanent=True),
        ).add_to(fmap)
    fmap.save(str(out))


def print_table(results: list[Result]) -> None:
    rows = [
        {
            "segment_key": r.key, "road name": r.osm_road_name or "-", "highway": r.highway or "-",
            "frc": r.frc or "-", "currentSpeed": r.current_speed if r.current_speed is not None else "-",
            "freeFlowSpeed": r.free_flow_speed if r.free_flow_speed is not None else "-",
            "conf": r.confidence if r.confidence is not None else "-",
            "snap_m": r.snap_m if r.snap_m is not None else "-", "status": r.status,
        }
        for r in results
    ]
    print(pd.DataFrame(rows).to_string(index=False))


# --- pipeline --------------------------------------------------------------------------

def process_location(
    key: str, name: str, geo: tuple[float, float] | None, ref: tuple[float, float],
    edges: gpd.GeoDataFrame, api_key: str,
) -> Result:
    """Geocode result -> distance check -> snap -> TomTom validation, never guessing."""
    res = Result(key=key, name=name)
    if geo is None:
        res.status = "FAILED: geocode"
        return res
    dist_km = ox.distance.great_circle(geo[0], geo[1], ref[0], ref[1]) / 1000
    if dist_km > MAX_DIST_KM:
        res.status = f"FAILED: {dist_km:.1f} km from Silk Board"
        return res
    snap = snap_to_edge(geo[0], geo[1], edges)
    if snap is None:
        res.status = "FAILED: no major edge"
        return res
    if snap["snap_m"] > MAX_SNAP_M:
        res.snap_m = snap["snap_m"]
        res.status = f"FAILED: snap {snap['snap_m']:.0f} m > {MAX_SNAP_M:.0f} m"
        return res
    res.lat, res.lon, res.osm_road_name = snap["lat"], snap["lon"], snap["osm_road_name"]
    res.highway, res.snap_m = snap["highway"], snap["snap_m"]
    flow = tomtom_flow(res.lat, res.lon, api_key)
    if "error" in flow:
        res.status = f"FAILED: {flow['error']}"
        return res
    res.frc, res.current_speed = flow["frc"], flow["current_speed"]
    res.free_flow_speed, res.confidence = flow["free_flow_speed"], flow["confidence"]
    return res


def main() -> int:
    api_key = get_env("TOMTOM_API_KEY")
    if not api_key:
        log.error("TOMTOM_API_KEY is not set (put it in .env)")
        return 1

    ox.settings.http_user_agent = USER_AGENT
    ox.settings.http_referer = USER_AGENT
    ox.settings.cache_folder = str(zone_dir("staging") / "osmnx_cache")

    cache_path = zone_dir("staging") / "geocode_cache.json"
    cache = _load_cache(cache_path)
    geocoded = {key: geocode(name, cache, cache_path) for key, name in LOCATIONS}

    ref = geocoded.get(REFERENCE_KEY)
    if ref is None:
        log.error("cannot geocode the reference point (Silk Board); nothing can be validated")
        return 1

    edges = load_major_edges(ref, radius_m=(MAX_DIST_KM + 0.5) * 1000)
    log.info("loaded %d major-road edges around Silk Board", len(edges))

    results = [process_location(k, n, geocoded[k], ref, edges, api_key) for k, n in LOCATIONS]
    good = [r for r in results if r.ok]

    if good:
        write_segments(SETTINGS_PATH, [_segment_entry(r) for r in good])
        build_map(results, zone_dir("reports") / "figures" / "segments_map.html")
        log.info("wrote %d segments to %s", len(good), SETTINGS_PATH.name)

    print_table(results)
    failed = [r for r in results if not r.ok]
    if failed:
        print("\nFAILED -- fix manually:")
        for r in failed:
            print(f"  {r.key} ({r.name}): {r.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
