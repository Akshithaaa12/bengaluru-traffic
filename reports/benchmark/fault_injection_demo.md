# Fault-injection demo

Two **optional** sources are broken on purpose while the pipeline runs on a 14-day slice of METR-LA
(40 sensors, 161,280 rows). Nothing in the real raw or quarantine folders is touched.

| | Fault | Source |
|---|---|---|
| (a) | Weather API failing (HTTP 503) for 3 days: 2012-03-05 to 2012-03-07 | `metr_weather` |
| (b) | Calendar file missing | `metr_calendar` |

## Which source failed and how it was detected

| Source | Detection | Evidence |
|---|---|---|
| metr_weather | Retry x3 with backoff exhausted, then quarantined (R4) | quarantine record: `error_type=HTTP_503`, `http_status=503`, `attempts=4`, `status=quarantined` |
| metr_weather | Null hourly values in the saved file (R5) | 72 hours null per variable (temperature_2m: 72, precipitation: 72, relative_humidity_2m: 72) |
| metr_weather | No valid weather for the timestamp after the hour join (R6) | `weather_available=0` on 864 timestamps |
| metr_calendar | Calendar file not found (R10) | `holiday_available=0` on 161,280 rows (all) |

## Flags set

| Flag | Rows | Meaning |
|---|---|---|
| `weather_available=0` | 34,560 (baseline: 0) | no valid weather; weather features NaN (forward-fill <= 1 h only) |
| `holiday_available=0` | 161,280 | calendar unavailable; `is_holiday` defaulted to 0 (rare event, flagged, not silently filled) |
| `sensor_missing`, `interpolated` | unchanged: True | the speed source was unaffected |

## Outcome

- The pipeline **completed without raising**; no crash from the missing weather window or the missing calendar.
- Usable modelling rows: 147,279 without faults, 116,937 with faults
  (30,342 rows excluded because required weather features were NaN; rows are
  excluded, not guessed).
- The failure is traceable: the quarantine record names the source, error and attempts; the curated rows carry the flags.
