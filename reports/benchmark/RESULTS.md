# METR-LA benchmark - results

## Problem

Predict the congestion level (**Low** speed_ratio >= 0.75, **Moderate** 0.50-0.75, **Severe** < 0.50, where
`speed_ratio = speed / free_flow`, free flow = the sensor's 95th-percentile valid speed) **30 minutes ahead** for 40
randomly sampled METR-LA sensors (`random_state=42`), then compare shortest-distance routing with congestion-aware
routing. This benchmark validates the pipeline on a public dataset while Bengaluru data is still being collected.

## Pipeline

**Problem -> Sources -> Ingestion -> Raw -> Transform -> Integrate -> Curated -> ML**

1. *Problem:* congestion class 30 min ahead, then congestion-aware routing.
2. *Sources:* three independent sources with different formats and keys (table below); the Bengaluru TomTom collector runs alongside.
3. *Ingestion:* file download, REST API with retry x3 + quarantine, and a library; every raw zone has a `_manifest.json`.
4. *Raw:* append-only, one folder per source, never merged.
5. *Transform:* apply the DQ rules (speed 0 = failure, flags, interpolation <= 30 min, weather forward-fill <= 1 h).
6. *Integrate:* join on `(sensor_id, timestamp_5min)`, weather on the hour, holidays on the date.
7. *Curated:* `data/curated/metr_curated.parquet` (every row, with flags, a documented source per column: `data_dictionary.csv`).
8. *ML:* features -> persistence / LR / RF / XGBoost -> SHAP -> routing.

## Data sources (kept separate in raw zones)

| source | role | format | ingestion | raw_zone | integration_key | why_useful |
|---|---|---|---|---|---|---|
| Bengaluru TomTom live collector | Primary live source | JSON (gzip) | REST API polled every 15 min (GitHub Actions + cron-job.org) | data/raw/traffic_api/ | (segment_key, timestamp_15min) | Primary source: live congestion for the real target city (13 South-East Bengaluru segments) (6 raw run files so far) |
| METR-LA sensors | Benchmark training source | HDF5 matrix + CSV + TXT | File download (zip) | data/raw/metr_la/ | (sensor_id, timestamp_5min) | Benchmark training data: 4 months of loop-detector speeds to validate the model (no public Indian sensor history exists); locations/distances build the routing graph |
| Open-Meteo weather | Benchmark feature source | JSON (hourly arrays) | REST API, retry x3 + quarantine | data/raw/metr_weather/ | hour (timestamp floored to 1 h) | Rain/temperature/humidity can change speeds and congestion |
| US holidays | Benchmark feature source | CSV | `holidays` library | data/raw/metr_calendar/ | date | Holiday traffic differs from normal weekdays |

### Schema differences

The sources share nothing except time: METR-LA is a wide HDF5 matrix (timestamp x 207 sensor columns) plus CSVs, Open-Meteo is
column-oriented JSON arrays with ISO time strings, and the holiday list is a two-column CSV. They have different granularity
(5 min, 1 h, 1 day), so each joins on a different key (see [`source_schemas.md`](source_schemas.md)).

## Data-quality rules

Mandatory fields (`sensor_id`, `timestamp`, `speed`) must be present or the record/sensor is rejected or excluded; optional
sources (weather, holiday) only enrich a row and are flagged when missing (`weather_available`).

| rule_id | source | field | requirement | detection | handling |
|---|---|---|---|---|---|
| R1 | metr_la | speed | mandatory | speed == 0 (or NaN): a loop detector reporting 0 mph is a sensor failure, never a real zero | flag sensor_missing=1 and set NaN; interpolate gaps <= 30 min (interpolated=1); exclude longer gaps |
| R2 | metr_la | timestamp | mandatory | null, duplicated or not on the regular 5-min grid | reject the record (cannot be placed in time) |
| R3 | metr_la | sensor_id | mandatory | missing, or no entry in sensor_locations | exclude the sensor |
| R4 | metr_weather | api_request | optional | timeout, connection error, HTTP 429 or 5xx | retry x3 with exponential backoff, then quarantine; no weather file -> NaN weather, weather_available=0 |
| R5 | metr_weather | temperature_2m, precipitation, relative_humidity_2m | optional | null hourly value | weather_available=0; forward-fill <= 1 h only; rows still missing are excluded from modelling |
| R6 | metr_weather | join_on_hour | optional | no weather row for floor(timestamp, 1 h) | same as R5 |
| R7 | metr_calendar | holiday | optional | date absent from the holiday list | valid is_holiday=0 (not missing) |
| R9 | metr_la, metr_weather | speed, temperature_2m, relative_humidity_2m, precipitation | mandatory (speed) / optional (weather) | range check before integration: speed 0-100 mph, temperature -20..55 C, humidity 0-100 %, precipitation >= 0; timestamps strictly increasing; no duplicate (sensor_id, timestamp) | out-of-range value -> NaN + speed_out_of_range / weather_out_of_range flag; duplicate/unsorted timestamps -> sorted, first duplicate kept |
| R10 | metr_calendar | calendar_file | optional | holiday file missing | holiday_available=0 and is_holiday=0 (justified default: rare event, flagged) |
| R8 | integration | features + target | derived | NaN in any required feature or the 30-min-ahead target (lags/rolling/target touch a long gap) | exclude the row from train/val/test (kept in the curated table with usable=0) |

## Missing data by source

| source | check | requirement | total_records | missing_or_failed | pct | detected_by | handling | filled | excluded |
|---|---|---|---|---|---|---|---|---|---|
| metr_la | speed == 0, all 207 sensors | mandatory | 7,094,304 | 575,302 | 8.110 | R1: speed == 0 | flag + set NaN | - | - |
| metr_la | speed == 0 / NaN, 40 selected sensors | mandatory | 1,370,880 | 107,173 | 7.820 | R1: speed == 0 or NaN | flag sensor_missing=1; interpolate gaps <= 30 min; exclude longer | 6,470 interpolated | 100,703 excluded |
| metr_la | speed outside 0-100 mph | mandatory | 7,094,304 | 0 | 0.000 | R9: range check before integration | set NaN + speed_out_of_range=1 (then handled as R1) | - | - |
| metr_la | non-monotonic / duplicate (sensor_id, timestamp) | mandatory | 34,272 | 0 | 0.000 | R9: strictly increasing index, duplicated keys | sort, keep first duplicate | - | - |
| metr_la | timestamp null / duplicated / irregular | mandatory | 34,272 | 0 | 0.000 | R2: isnull, duplicated, step != 5 min | reject record | - | 0 rejected |
| metr_la | sensor without coordinates | mandatory | 207 | 0 | 0.000 | R3: no row in sensor_locations | exclude sensor | - | 0 excluded |
| metr_weather | API request failed | optional | 1 | 0 | 0.000 | R4: timeout / 429 / 5xx | retry x3 then quarantine; weather_available=0 | - | - |
| metr_weather | temperature_2m null hours | optional | 2,856 | 0 | 0.000 | R5: null hourly value | weather_available=0; ffill <= 1 h; else exclude | - | - |
| metr_weather | precipitation null hours | optional | 2,856 | 0 | 0.000 | R5: null hourly value | weather_available=0; ffill <= 1 h; else exclude | - | - |
| metr_weather | relative_humidity_2m null hours | optional | 2,856 | 0 | 0.000 | R5: null hourly value | weather_available=0; ffill <= 1 h; else exclude | - | - |
| metr_weather | temperature_2m out of range | optional | 2,856 | 0 | 0.000 | R9: range check before integration | set NaN + weather_out_of_range=1; weather_available=0 | - | - |
| metr_weather | precipitation out of range | optional | 2,856 | 0 | 0.000 | R9: range check before integration | set NaN + weather_out_of_range=1; weather_available=0 | - | - |
| metr_weather | relative_humidity_2m out of range | optional | 2,856 | 0 | 0.000 | R9: range check before integration | set NaN + weather_out_of_range=1; weather_available=0 | - | - |
| metr_weather | timestamps with no weather after the hour join | optional | 34,272 | 0 | 0.000 | R6: no row for floor(timestamp, 1 h) | weather_available=0; ffill <= 1 h; else exclude | - | - |
| metr_calendar | date not a holiday | optional | 34,272 | 0 | 0.000 | R7: absent date | valid is_holiday=0 (not missing) | - | - |
| metr_calendar | calendar file unavailable (rows with holiday_available=0) | optional | 34,272 | 0 | 0.000 | R10: file missing | holiday_available=0, is_holiday=0 (flagged default) | - | - |
| integration | rows with a NaN required feature or target | derived | 1,370,880 | 132,902 | 9.690 | R8: NaN in features/target | exclude from modelling; keep in curated with usable=0 | - | 132,902 excluded |

Full detail: [`dq_summary.md`](dq_summary.md), [`missing_by_source.csv`](missing_by_source.csv).

### Validation before integration (R9)

Each source is validated on its own before any join: speed must be within 0-100 mph, temperature -20..55 C, humidity 0-100 %,
precipitation >= 0; timestamps must be strictly increasing with no duplicate `(sensor_id, timestamp)`. Out-of-range values become
NaN and are flagged (`speed_out_of_range`, `weather_out_of_range`); unsorted/duplicated keys are sorted and de-duplicated (first
kept). On this data every check found 0 violations (see the table above).

### Failure handling demonstrated

[`fault_injection_demo.md`](fault_injection_demo.md) breaks two optional sources on purpose (a 3-day weather API outage and a missing
calendar file) and shows which source failed, how it was detected (quarantine record, null counts, `weather_available=0`,
`holiday_available=0`), the flags set, and that the run still completes. Every curated file also stores the manifest path and
`ingested_at` of each raw source in its metadata, so a flagged row can be traced back to its source.

### Timezone assumption

METR-LA timestamps carry no timezone. They are **treated as America/Los_Angeles local time**, and the Open-Meteo request uses
`timezone=America/Los_Angeles`, so the hourly weather join, hour-of-day, weekday and peak features are all in local time. Timestamps
are stored naive (not UTC, unlike the Bengaluru TomTom pipeline). Daylight-saving transitions (2012-03-11) were not verified against
the sensor data.

## Models and metrics (test set, chronological 70/15/15 split)

Features: speed-ratio lags (5/15/30/60 min), 1 h rolling mean/std, sin/cos hour and weekday, peak flag, holiday flag,
temperature, precipitation, humidity and the sensor's free-flow speed. The final model is **XGBoost** (class-weighted),
chosen because Severe-class recall matters most.

| model | n | accuracy | macro_f1 | recall_low | recall_moderate | recall_severe |
|---|---|---|---|---|---|---|
| persistence | 172,519 | 0.883 | 0.675 | 0.951 | 0.417 | 0.659 |
| logistic_regression | 172,519 | 0.885 | 0.599 | 0.977 | 0.121 | 0.697 |
| random_forest | 172,519 | 0.899 | 0.689 | 0.977 | 0.343 | 0.666 |
| xgboost | 172,519 | 0.847 | 0.683 | 0.877 | 0.649 | 0.743 |

XGBoost reaches Severe recall 0.743 versus 0.659 for persistence, but with lower
accuracy (0.847 vs 0.883) because class weighting trades Low-class recall for Moderate/Severe
recall. Confusion matrix: `confusion_matrix.png`. Validation metrics are in `metrics.csv`.

## SHAP - top features for the Severe class

TreeExplainer on XGBoost, 2,000 random test rows (`shap_summary.png`, `shap_bar.png`, `shap_top_features.csv`).

| feature | mean_abs_shap_severe | share_severe |
|---|---|---|
| hour_cos | 0.665 | 25.5% |
| ratio_t | 0.657 | 25.2% |
| free_flow_mph | 0.331 | 12.7% |
| hour_sin | 0.198 | 7.6% |
| ratio_lag5 | 0.180 | 6.9% |
| ratio_std_1h | 0.086 | 3.3% |
| wd_sin | 0.075 | 2.9% |
| temp_c | 0.067 | 2.6% |
| wd_cos | 0.065 | 2.5% |
| humidity | 0.058 | 2.2% |

## What each source contributes (ablation)

XGBoost retrained with feature groups removed (same splits and hyperparameters; the final model is unchanged). Test set:

| variant | n_features | test_accuracy | test_macro_f1 | test_recall_severe | delta_macro_f1 | delta_recall_severe |
|---|---|---|---|---|---|---|
| all_features | 17 | 0.8471 | 0.6827 | 0.7432 | 0.0000 | 0.0000 |
| no_weather | 14 | 0.8486 | 0.6846 | 0.7479 | 0.0019 | 0.0046 |
| no_holiday | 16 | 0.8464 | 0.6825 | 0.7480 | -0.0002 | 0.0048 |
| no_free_flow_mph | 16 | 0.8354 | 0.6629 | 0.7172 | -0.0199 | -0.0260 |
| no_weather_no_holiday | 13 | 0.8482 | 0.6839 | 0.7488 | 0.0012 | 0.0055 |

- **Weather** (temperature, precipitation, humidity): without it macro-F1 0.685 (+0.002), Severe recall 0.748 (+0.005) - no measurable contribution. It is a single
  Los Angeles point shared by all 40 sensors, and rain is rare in this period.
- **Holidays**: without them macro-F1 0.682 (-0.000), Severe recall 0.748 (+0.005) - no measurable contribution. The 119-day range contains one holiday (Memorial Day).
- **Both optional sources removed**: macro-F1 0.684 (+0.001), Severe recall 0.749 (+0.006).
- **`free_flow_mph`**: without it macro-F1 0.663 (-0.020), Severe recall 0.717 (-0.026) - a real contribution; it carries per-sensor information (it acts
  as a sensor identifier).

Honest reading: on METR-LA the predictive signal comes from the speed history of the sensors. Weather and holidays are integrated
as genuinely separate sources to demonstrate the multi-source pipeline, but they do **not** improve these metrics here; whether they matter
more in Bengaluru (for example monsoon rain) is untested. Differences of a few thousandths come from a single run without confidence
intervals and should not be over-interpreted.

## Experiment tracking (MLflow)

Every model (persistence, logistic regression, random forest, XGBoost) and every ablation variant is logged to a local MLflow store
(`./mlruns`, experiment `metr_la_benchmark`): parameters, validation/test metrics and artifacts (confusion matrix; SHAP summary and bar
plots for the XGBoost runs). Run `mlflow ui` to browse them. MLflow 3 marks the file store as maintenance mode, so it is enabled with
`MLFLOW_ALLOW_FILE_STORE=true` (set in `src/benchmark/tracking.py`).

## Routing

Directed graph of 19 sensors / 68 edges (road distance < 5 km, largest weakly connected component of
the 40 sampled sensors). Edge time = distance / speed at the destination sensor. For 50 random O-D pairs at the
weekday 08:00 test timestamp 2012-06-11 08:00:00, the congestion-aware route is chosen with XGBoost-predicted speeds
and both routes are timed with the **actual** speeds at 2012-06-11 08:30:00.

- **Mean saving: 2.79%, median saving: 0.25%**
- The two routes differ in 27 of 50 pairs (mean saving 5.17% over those);
  26 pairs improved, 1 got worse.

Output: `routing_results.csv`, `routing_example.png` (the example shown is the best of the 50 pairs, not a typical one).

## Limitations

- **Not Bengaluru.** Models are trained and tested on Los Angeles 2012 data; Bengaluru collection only started on 2026-10-07.
- **XGBoost is not clearly better than persistence overall.** Accuracy is lower (0.847 vs 0.883) and macro-F1
  only marginally higher (0.683 vs 0.675); its advantage is Severe/Moderate recall, bought with more false
  alarms. It also used all 300 boosting rounds (early-stopping cap), so it is probably under-trained.
- **Possible shortcuts.** `hour_cos` and `free_flow_mph` are top SHAP features; `free_flow_mph` acts as a sensor ID and train/test use the
  same 40 sensors, so spatial generalisation is untested.
- **Mild leakage.** The 95th-percentile free-flow speed is computed over the whole period, including validation/test.
- **Small sample.** 40 of 207 sensors; SHAP on 2,000 rows; metrics have no confidence intervals.
- **Routing is a simplification.** One timestamp, one 19-sensor component, a sensor graph rather than a road network, edge time from the
  destination sensor only, and predicted speeds derived from class probabilities x class-mean ratios (the model predicts classes, not speeds).
  The mean saving is small and the median is close to zero; large savings occur only on a few pairs.
- **Time zone.** Naive METR-LA timestamps are treated as America/Los_Angeles local time and stored without UTC conversion;
  daylight-saving handling is unverified.
- **Optional sources add little here.** The ablation shows weather and holidays do not improve the metrics on this benchmark.
- **Test coverage.** Unit tests cover the DQ rules, validation, weather retry/quarantine, the TomTom collector and the fault demo;
  the training, SHAP and routing code have no unit tests.
