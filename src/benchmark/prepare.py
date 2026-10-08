"""Clean METR-LA, label congestion and build the 30-min-ahead feature table."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark.download import CALENDAR_DIR, H5_PATH, WEATHER_DIR
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
    sensor_missing = raw.eq(0) | raw.isna()
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


def load_weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly weather aligned to ``index`` (floored to the hour); forward-fill <= 1 h only."""
    path = next(WEATHER_DIR.glob("open_meteo_la_*.json"))
    hourly = json.loads(path.read_text())["response"]["hourly"]
    wx = pd.DataFrame(hourly).assign(time=lambda d: pd.to_datetime(d["time"])).set_index("time")
    hours = pd.date_range(index.min().floor("h"), index.max().floor("h"), freq="h")
    wx = wx.reindex(hours)
    available = wx["temperature_2m"].notna()
    wx = wx.ffill(limit=1)
    out = wx.reindex(index.floor("h"))
    out.index = index
    out["weather_available"] = available.reindex(index.floor("h")).to_numpy().astype(int)
    return out


def load_holiday_dates() -> pd.DatetimeIndex:
    df = pd.read_csv(next(CALENDAR_DIR.glob("us_holidays_*.csv")), parse_dates=["date"])
    return pd.DatetimeIndex(df["date"])


def time_features(index: pd.DatetimeIndex, holiday_dates: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour + index.minute / 60
    weekday = index.weekday
    peak = (weekday < 5) & (((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 20)))
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "wd_sin": np.sin(2 * np.pi * weekday / 7), "wd_cos": np.cos(2 * np.pi * weekday / 7),
        "is_peak": peak.astype(int), "is_holiday": index.normalize().isin(holiday_dates).astype(int),
    }, index=index)


def build_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (row-per-sensor-timestamp table, stats for the DQ summary)."""
    raw_all = load_raw_speeds()
    raw = pick_sensors(raw_all)
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
    tf = time_features(index, load_holiday_dates())
    wx = load_weather(index)

    cols: dict[str, np.ndarray] = {k: v.to_numpy(dtype="float32").ravel() for k, v in feats.items()}
    for name in tf.columns:
        cols[name] = np.repeat(tf[name].to_numpy(), n_s)
    for name, src in (("temp_c", "temperature_2m"), ("precip_mm", "precipitation"), ("humidity", "relative_humidity_2m")):
        cols[name] = np.repeat(wx[src].to_numpy(dtype="float32"), n_s)
    cols["free_flow_mph"] = np.tile(free_flow.to_numpy(dtype="float32"), n_t)
    cols["weather_available"] = np.repeat(wx["weather_available"].to_numpy(), n_s)
    cols["y_now"] = to_label(ratio_np).ravel()
    cols["y"] = target.to_numpy().ravel()
    cols["tpos"] = np.repeat(np.arange(n_t), n_s)
    cols["sensor"] = np.tile(np.arange(n_s), n_t)
    df = pd.DataFrame(cols)
    df["time"] = index[df["tpos"].to_numpy()]

    complete = df[FEATURES + ["y", "y_now"]].notna().all(axis=1)
    stats = {
        "raw_all_readings": int(raw_all.size), "raw_all_zero": int(raw_all.eq(0).to_numpy().sum()),
        "n_timestamps": n_t, "n_sensors": n_s, "start": str(index.min()), "end": str(index.max()),
        "readings": int(raw.size), "sensor_missing": int(c["sensor_missing"].to_numpy().sum()),
        "interpolated": int(c["interpolated"].to_numpy().sum()),
        "still_nan": int(c["filled"].isna().to_numpy().sum()),
        "weather_hours": int(wx["weather_available"].groupby(index.floor("h")).first().shape[0]),
        "weather_unavailable_rows": int((wx["weather_available"] == 0).sum()),
        "weather_missing_pct": {v: float(wx[v].isna().mean() * 100) for v in ("temperature_2m", "precipitation", "relative_humidity_2m")},
        "holidays_total": int(len(load_holiday_dates())),
        "holiday_rows": int(df["is_holiday"].sum()),
        "rows_total": int(len(df)), "rows_complete": int(complete.sum()),
    }
    return df[complete].reset_index(drop=True), stats


def chronological_split(df: pd.DataFrame, n_timestamps: int) -> dict[str, pd.DataFrame]:
    """70/15/15 by time. The last HORIZON steps of train/val are purged so targets don't cross splits."""
    i_train, i_val = int(0.70 * n_timestamps), int(0.85 * n_timestamps)
    t = df["tpos"]
    return {
        "train": df[t < i_train - HORIZON],
        "val": df[(t >= i_train) & (t < i_val - HORIZON)],
        "test": df[t >= i_val],
    }
