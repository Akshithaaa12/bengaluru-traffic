# METR-LA benchmark - results

## Problem

Predict the congestion level (**Low** speed_ratio >= 0.75, **Moderate** 0.50-0.75, **Severe** < 0.50, where
`speed_ratio = speed / free_flow`, free flow = the sensor's 95th-percentile valid speed) **30 minutes ahead** for 40
randomly sampled METR-LA sensors (`random_state=42`), then compare shortest-distance routing with congestion-aware
routing. This benchmark validates the pipeline on a public dataset while Bengaluru data is still being collected.

## Data sources (kept separate in raw zones)

| Source | Raw zone | Content |
|---|---|---|
| METR-LA sensors | `data/raw/metr_la/` | 5-min speeds (mph), 207 sensors, 2012-03-01 to 2012-06-27, sensor locations, road distances |
| Open-Meteo archive | `data/raw/metr_weather/` | Hourly temperature, precipitation, humidity for Los Angeles, joined on the hour |
| US holidays 2012 | `data/raw/metr_calendar/` | `holidays` library, joined on date |
| Bengaluru TomTom live collector | `data/raw/traffic_api/` | Flow Segment API, 13 validated South-East Bengaluru segments, every 15 min via GitHub Actions; 4 raw run files at the time of writing. **Not used for modelling yet** (too little history). |

## Missing-data handling

Speed 0 is a sensor failure, not a real zero: it is set to NaN and flagged `sensor_missing=1`. Gaps of 30 minutes or
less are linearly interpolated (`interpolated=1`); longer gaps stay NaN and those rows are excluded from training and
evaluation. Weather is forward-filled for at most 1 hour.

| Item | Count | % |
|---|---|---|
| Speed == 0 (all 207 sensors) | 575,302 | 8.11% |
| `sensor_missing=1` (40 sensors; speed 0 -> NaN) | 107,173 | 7.82% |
| `interpolated=1` (gap <= 30 min filled) | 6,470 | 0.47% |
| Still NaN after interpolation (gap > 30 min or at series edge; excluded) | 100,703 | 7.35% |
| `weather_available=0` timestamps | 0 | 0.00% |
| Weather `temperature_2m` still NaN after <=1 h ffill | - | 0.00% |
| Weather `precipitation` still NaN after <=1 h ffill | - | 0.00% |
| Weather `relative_humidity_2m` still NaN after <=1 h ffill | - | 0.00% |

Speed 0 is treated as a sensor failure, never as a real zero. Rows with any NaN feature or target (lags/rolling windows touching a long gap) are dropped before modelling.

Full detail: [`dq_summary.md`](dq_summary.md).

## Models and metrics (test set, chronological 70/15/15 split)

Features: speed-ratio lags (5/15/30/60 min), 1 h rolling mean/std, sin/cos hour and weekday, peak flag, holiday flag,
temperature, precipitation, humidity and the sensor's free-flow speed. The final model is **XGBoost** (class-weighted),
chosen because Severe-class recall matters most.

| model | n | accuracy | macro_f1 | recall_low | recall_moderate | recall_severe |
|---|---|---|---|---|---|---|
| persistence | 172519 | 0.883 | 0.675 | 0.951 | 0.417 | 0.659 |
| logistic_regression | 172519 | 0.885 | 0.599 | 0.977 | 0.121 | 0.697 |
| random_forest | 172519 | 0.899 | 0.689 | 0.977 | 0.343 | 0.666 |
| xgboost | 172519 | 0.847 | 0.683 | 0.877 | 0.649 | 0.743 |

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
- **Time zone.** Naive METR-LA timestamps are treated as America/Los_Angeles local time; daylight-saving handling is unverified.
- **No unit tests yet** for the benchmark modules.
