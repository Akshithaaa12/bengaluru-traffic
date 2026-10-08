"""Clean METR-LA, label congestion and build the 30-min-ahead feature table."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark.download import CALENDAR_DIR, H5_PATH, WEATHER_DIR
from src.benchmark.dq_rules import speed_failure_mask, validate_speeds, validate_weather
from src.utils.config import get_settings

N_SENSORS = 40
SEED = 42
STEP_MIN = 5
HORIZON = 6                  # 6 x 5 min = 30 min ahead
MAX_GAP_STEPS = 6            # interpolate gaps <= 30 min
LAGS = {5: 1, 15: 3, 30: 6, 60: 12}
ROLL = 12                    # 1 h
CLASSES = ["Low", "Moderate", "Severe"]
FEATURES = [
    "ratio_t", "ratio_lag5", "ratio_lag15", "ratio_lag30", "ratio_lag60",
    "ratio_mean_1h", "ratio_std_1h", "hour_sin", "hour_cos", "wd_sin", "wd_cos",
    "is_peak", "is_holiday", "temp_c", "precip_mm", "humidity", "free_flow_mph",
]


def load_raw_speeds() -> pd.DataFrame:
    """All 207 sensors, 5-min speeds in mph (0 = sensor failure)."""
    return pd.read_hdf(H5_PATH)


def pick_sensors(raw: pd.DataFrame) -> pd.DataFrame:
    cols = raw.columns.to_series().sample(n=N_SENSORS, random_state=SEED).tolist()
    return raw[cols]


def fill_short_gaps(col: pd.Series) -> pd.Series:
    """Linearly interpolate NaN runs <= MAX_GAP_STEPS; longer runs (and edge runs) stay NaN."""
    isna = col.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    return col.interpolate(limit_area="inside").where(~(isna & (run_len > MAX_GAP_STEPS)))


def clean_speeds(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Speed 0 is a sensor failure, not a real zero: NaN + flag, then short-gap interpolation."""
    sensor_missing = speed_failure_mask(raw)
    valid = raw.mask(sensor_missing)
    filled = valid.apply(fill_short_gaps)
    interpolated = valid.isna() & filled.notna()
    return {"valid": valid, "filled": filled, "sensor_missing": sensor_missing, "interpolated": interpolated}


def to_label(ratio: np.ndarray) -> np.ndarray:
    """0=Low (>=low), 1=Moderate, 2=Severe (<severe); NaN stays NaN."""
    t = get_settings()["thresholds"]
    out = np.full(ratio.shape, np.nan)
    out[ratio >= t["low"]] = 0
    out[(ratio < t["low"]) & (ratio >= t["severe"])] = 1
    out[ratio < t["severe"]] = 2
    return out


WEATHER_COLS = ["temperature_2m", "precipitation", "relative_humidity_2m"]


def load_weather(index: pd.DatetimeIndex, weather_dir: Path | None = None) -> pd.DataFrame:
    """Hourly weather aligned to ``index`` (floored to the hour); forward-fill <= 1 h only.

    Values are range-checked before the join (R9). If the raw weather file is missing (e.g. the API
    call was quarantined) every weather value is NaN and ``weather_available`` is 0 (R4).
    """
    files = sorted((weather_dir or WEATHER_DIR).glob("open_meteo_la_*.json"))
    if not files:
        out = pd.DataFrame(np.nan, index=index, columns=WEATHER_COLS)
        out["weather_available"], out["weather_out_of_range"] = 0, 0
        return out
    hourly = json.loads(files[0].read_text())["response"]["hourly"]
    wx = pd.DataFrame(hourly).assign(time=lambda d: pd.to_datetime(d["time"])).set_index("time")
    wx = wx.reindex(pd.date_range(index.min().floor("h"), index.max().floor("h"), freq="h"))
    wx, wx_flag, oor_counts = validate_weather(wx)
    available = wx[WEATHER_COLS].notna().all(axis=1)      # all three variables valid
    out = wx.ffill(limit=1).reindex(index.floor("h"))
    out.index = index
    out["weather_available"] = available.reindex(index.floor("h")).to_numpy().astype(int)
    out["weather_out_of_range"] = wx_flag.reindex(index.floor("h")).to_numpy().astype(int)
    out.attrs["out_of_range_counts"] = oor_counts
    return out


def load_holiday_dates(calendar_dir: Path | None = None) -> pd.DatetimeIndex | None:
    """Holiday dates from the raw calendar file, or None if the file is missing (R10)."""
    files = sorted((calendar_dir or CALENDAR_DIR).glob("us_holidays_*.csv"))
    if not files:
        return None
    return pd.DatetimeIndex(pd.read_csv(files[0], parse_dates=["date"])["date"])


def time_features(index: pd.DatetimeIndex, holiday_dates: pd.DatetimeIndex | None) -> pd.DataFrame:
    hour = index.hour + index.minute / 60
    weekday = index.weekday
    peak = (weekday < 5) & (((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 20)))
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "wd_sin": np.sin(2 * np.pi * weekday / 7), "wd_cos": np.cos(2 * np.pi * weekday / 7),
        "is_peak": peak.astype(int),
        "is_holiday": (index.normalize().isin(holiday_dates).astype(int) if holiday_dates is not None else 0),
        "holiday_available": int(holiday_dates is not None),
    }, index=index)


def build_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (modelling rows only, stats for the DQ summary)."""
    full, stats = build_full_table()
    return full[full["usable"] == 1].reset_index(drop=True), stats


def build_full_table(
    raw_all: pd.DataFrame | None = None, weather_dir: Path | None = None, calendar_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Integrated row-per-sensor-timestamp table (all rows, with flags) and stats for the DQ summary.

    Each source is validated/cleaned on its own first (speed keys + ranges, weather ranges, calendar
    availability) and only then joined. The arguments let tests and the fault demo inject inputs.
    """
    raw_all = load_raw_speeds() if raw_all is None else raw_all
    checked = validate_speeds(raw_all)                       # R9, before anything else
    raw = pick_sensors(checked["speeds"])
    speed_oor = checked["speed_out_of_range"][raw.columns]
    c = clean_speeds(raw)

    free_flow = c["valid"].quantile(0.95)                  # per sensor, valid speeds only
    ratio = c["filled"] / free_flow
    ratio_np = ratio.to_numpy(dtype="float64")
    feats = {"ratio_t": ratio}
    for minutes, steps in LAGS.items():
        feats[f"ratio_lag{minutes}"] = ratio.shift(steps)
    roll = ratio.rolling(ROLL, min_periods=ROLL)
    feats["ratio_mean_1h"], feats["ratio_std_1h"] = roll.mean(), roll.std()
    target = pd.DataFrame(to_label(ratio_np), index=ratio.index, columns=ratio.columns).shift(-HORIZON)

    index = ratio.index
    n_t, n_s = ratio.shape
    holiday_dates = load_holiday_dates(calendar_dir)
    tf = time_features(index, holiday_dates)
    wx = load_weather(index, weather_dir)

    cols: dict[str, np.ndarray] = {k: v.to_numpy(dtype="float32").ravel() for k, v in feats.items()}
    for name in tf.columns:
        cols[name] = np.repeat(tf[name].to_numpy(), n_s)
    for name, src in (("temp_c", "temperature_2m"), ("precip_mm", "precipitation"), ("humidity", "relative_humidity_2m")):
        cols[name] = np.repeat(wx[src].to_numpy(dtype="float32"), n_s)
    cols["free_flow_mph"] = np.tile(free_flow.to_numpy(dtype="float32"), n_t)
    cols["weather_available"] = np.repeat(wx["weather_available"].to_numpy(), n_s)
    cols["weather_out_of_range"] = np.repeat(wx["weather_out_of_range"].to_numpy(), n_s)
    cols["speed_out_of_range"] = speed_oor.to_numpy(dtype="int8").ravel()
    cols["y_now"] = to_label(ratio_np).ravel()
    cols["y"] = target.to_numpy().ravel()
    cols["speed_mph"] = c["valid"].to_numpy(dtype="float32").ravel()
    cols["speed_filled_mph"] = c["filled"].to_numpy(dtype="float32").ravel()
    cols["sensor_missing"] = c["sensor_missing"].to_numpy(dtype="int8").ravel()
    cols["interpolated"] = c["interpolated"].to_numpy(dtype="int8").ravel()
    cols["sensor_id"] = np.tile(ratio.columns.astype(str).to_numpy(), n_t)
    cols["tpos"] = np.repeat(np.arange(n_t), n_s)
    cols["sensor"] = np.tile(np.arange(n_s), n_t)
    df = pd.DataFrame(cols)
    df["time"] = index[df["tpos"].to_numpy()]

    complete = df[FEATURES + ["y", "y_now"]].notna().all(axis=1)
    df["usable"] = complete.astype("int8")
    stats = {
        "raw_all_readings": int(raw_all.size), "raw_all_zero": int(raw_all.eq(0).to_numpy().sum()),
        "speed_out_of_range": int(checked["speed_out_of_range"].to_numpy().sum()), "index_issues": checked["index_counts"],
        "weather_out_of_range": wx.attrs.get("out_of_range_counts", {}),
        "holiday_unavailable_rows": int((tf["holiday_available"] == 0).sum()),
        "n_timestamps": n_t, "n_sensors": n_s, "start": str(index.min()), "end": str(index.max()),
        "readings": int(raw.size), "sensor_missing": int(c["sensor_missing"].to_numpy().sum()),
        "interpolated": int(c["interpolated"].to_numpy().sum()),
        "still_nan": int(c["filled"].isna().to_numpy().sum()),
        "weather_hours": int(wx["weather_available"].groupby(index.floor("h")).first().shape[0]),
        "weather_unavailable_rows": int((wx["weather_available"] == 0).sum()),
        "weather_missing_pct": {v: float(wx[v].isna().mean() * 100) for v in WEATHER_COLS},
        "holidays_total": 0 if holiday_dates is None else int(len(holiday_dates)),
        "holiday_rows": int(df["is_holiday"].sum()),
        "rows_total": int(len(df)), "rows_complete": int(complete.sum()),
    }
    return df, stats


def chronological_split(df: pd.DataFrame, n_timestamps: int) -> dict[str, pd.DataFrame]:
    """70/15/15 by time. The last HORIZON steps of train/val are purged so targets don't cross splits."""
    i_train, i_val = int(0.70 * n_timestamps), int(0.85 * n_timestamps)
    t = df["tpos"]
    return {
        "train": df[t < i_train - HORIZON],
        "val": df[(t >= i_train) & (t < i_val - HORIZON)],
        "test": df[t >= i_val],
    }
