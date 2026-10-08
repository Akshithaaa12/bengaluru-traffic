# Source schemas and integration

The sources differ in format, granularity and key; they are kept separate in raw zones and only joined in the curated table.

| Source | Format | Granularity | Integration key |
|---|---|---|---|
| METR-LA sensors | HDF5 matrix + CSV + TXT | 5 min | (sensor_id, timestamp_5min) |
| Open-Meteo weather | JSON (hourly arrays) | 1 h | hour (timestamp floored to 1 h) |
| US holidays | CSV | 1 day | date |
| Bengaluru TomTom live collector | JSON (gzip) | 15 min | (segment_key, timestamp_15min) |

## metr_la

Format: HDF5 (speeds, wide matrix) + CSV (locations, distances) + TXT. Ingestion: file download (HTTP GET of a zip), unzipped unchanged.

**`metr_la.h5`**

| Column | Type / unit |
|---|---|
| `index` | timestamp (datetime64[ns]) |
| `columns` | 207 sensor_id columns |
| `value_dtype` | float32 |
| `unit` | mph; 0 = sensor failure |

**`sensor_locations_la.csv`**

| Column | Type / unit |
|---|---|
| `index` | int64 |
| `sensor_id` | int64 |
| `latitude` | float64 |
| `longitude` | float64 |

**`distances_la.csv`**

| Column | Type / unit |
|---|---|
| `from` | int64 |
| `to` | int64 |
| `cost` | float64 |

- **Integration key:** (sensor_id, timestamp_5min)
- **Features contributed:** speed_mph, speed_filled_mph, free_flow_mph, ratio_t, ratio_lag5/15/30/60, ratio_mean_1h, ratio_std_1h, sensor_missing, interpolated, y_now, y; timestamp-derived hour_sin/cos, wd_sin/cos, is_peak

## metr_weather

Format: JSON (column-oriented hourly arrays). Ingestion: REST API (Open-Meteo archive), retry x3 + quarantine.

**`open_meteo_la_20120301_20120627.json`**

| Column | Type / unit |
|---|---|
| `time` | string (ISO local time) |
| `temperature_2m` | float (°C) |
| `precipitation` | float (mm) |
| `relative_humidity_2m` | float (%) |

- **Integration key:** hour = floor(timestamp, 1 h)
- **Features contributed:** temp_c, precip_mm, humidity, weather_available

## metr_calendar

Format: CSV. Ingestion: python library `holidays` 0.106 (US, 2012).

**`us_holidays_2012.csv`**

| Column | Type / unit |
|---|---|
| `date` | str |
| `name` | str |
| `source` | str |
| `schema_version` | int64 |
| `ingested_at` | str |

- **Integration key:** date = timestamp.date
- **Features contributed:** is_holiday

## Curated dataset (`data/curated/metr_curated.parquet`) - source of every column

| Column | Source | Group | Description |
|---|---|---|---|
| `sensor_id` | metr_la | key | Loop-detector id (integration key part 1) |
| `timestamp` | metr_la | key | 5-min timestamp, naive local time (integration key part 2) |
| `speed_mph` | metr_la | speed | Valid speed; NaN where the sensor failed (speed 0) |
| `speed_filled_mph` | metr_la | speed | Speed after interpolating gaps <= 30 min; NaN for longer gaps |
| `sensor_missing` | metr_la | dq_flag | 1 if the raw speed was 0/NaN (sensor failure) |
| `interpolated` | metr_la | dq_flag | 1 if the value was filled by interpolation (gap <= 30 min) |
| `speed_out_of_range` | metr_la | dq_flag | 1 if the raw speed was outside 0-100 mph (set NaN before cleaning) |
| `free_flow_mph` | metr_la | speed | Sensor's 95th-percentile valid speed |
| `ratio_t` | metr_la | speed_lags | speed / free_flow now |
| `ratio_lag5` | metr_la | speed_lags | ratio 5 min ago |
| `ratio_lag15` | metr_la | speed_lags | ratio 15 min ago |
| `ratio_lag30` | metr_la | speed_lags | ratio 30 min ago |
| `ratio_lag60` | metr_la | speed_lags | ratio 60 min ago |
| `ratio_mean_1h` | metr_la | speed_rolling | Rolling 1 h mean of ratio |
| `ratio_std_1h` | metr_la | speed_rolling | Rolling 1 h std of ratio |
| `hour_sin` | metr_la (timestamp) | calendar_time | sin of hour of day |
| `hour_cos` | metr_la (timestamp) | calendar_time | cos of hour of day |
| `wd_sin` | metr_la (timestamp) | calendar_time | sin of weekday |
| `wd_cos` | metr_la (timestamp) | calendar_time | cos of weekday |
| `is_peak` | metr_la (timestamp) | calendar_time | Weekday 07-10 or 16-20 |
| `temp_c` | metr_weather | weather | Temperature 2 m (C), joined on the hour |
| `precip_mm` | metr_weather | weather | Precipitation (mm), joined on the hour |
| `humidity` | metr_weather | weather | Relative humidity 2 m (%), joined on the hour |
| `weather_available` | metr_weather | dq_flag | 1 if a valid hourly weather row existed for this timestamp |
| `weather_out_of_range` | metr_weather | dq_flag | 1 if a weather value was out of range (set NaN) |
| `holiday_available` | metr_calendar | dq_flag | 1 if the calendar file was available (0 -> is_holiday defaulted to 0) |
| `is_holiday` | metr_calendar | holiday | 1 if the date is a US holiday (joined on date); no match = valid 0 |
| `y_now` | metr_la | target | Congestion class now (0 Low, 1 Moderate, 2 Severe) |
| `y` | metr_la | target | Congestion class 30 min ahead (target) |
| `usable` | integration | dq_flag | 1 if all required features and the target are present (used for modelling) |
