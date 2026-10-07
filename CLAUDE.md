# Bengaluru Traffic Congestion Prediction & Route Optimization

ML-II course project (CMR Institute of Technology). Author: Akshitha.
Must follow the professor's "Data Sources + Missing Data Handling" guideline (multi-source pipeline).

## Problem
Predict congestion level (Low / Moderate / Severe) for Bengaluru road segments 15–60 min ahead,
then recommend congestion-aware routes (Dijkstra/A*) vs plain shortest path. Explain with SHAP.

Label: `speed_ratio = current_speed / free_flow_speed`
- Low: ratio >= 0.75 · Moderate: 0.50–0.75 · Severe: < 0.50  (thresholds in config, tune after EDA)

Study area: South-East Bengaluru (Koramangala, HSR, Silk Board, ORR–Bellandur, Marathahalli, Sarjapur Rd,
Electronic City corridor). Dense network so alternative routes exist.

## Data sources (keep segregated — never merge raw data)
| Source | Type | Ingestion | Raw zone |
|---|---|---|---|
| TomTom Traffic Flow Segment API | JSON, near-real-time | Scheduled polling every 15 min (GitHub Actions) | `data/raw/traffic_api/` |
| Open-Meteo weather (archive + forecast) | JSON time series | API | `data/raw/weather/` |
| OpenStreetMap road network | Geospatial graph | OSMnx → GraphML/GeoJSON | `data/raw/road_network/` |
| Bengaluru Traffic Police (OpenCity) | CSV (crashes, one-way roads) + PDF (signal timings) | File + document parsing | `data/raw/btp/` |
| Karnataka holidays | Reference | `holidays` lib → CSV | `data/raw/calendar/` |

Every raw record carries metadata: `source`, `ingested_at` (UTC), `source_file`/`request_url`, `schema_version`, `status`.
Raw files are append-only and never edited. Failed API calls are logged to `data/quarantine/` with the error.

## Integration
- Common key: `(segment_id, timestamp_15min)`. `segment_id` = OSM edge id (u, v, key) matched from the TomTom point.
- Weather → joined on the hour. Holidays → on date. BTP crashes → aggregated per segment (spatial join, buffer).
- One-way roads → applied to the routing graph, not to the feature table.

## Missing-data rules (document every decision)
- Never fill NULL with 0 by default. Find out why it's missing first.
- TomTom call failed / rate-limited → retry ×3, then quarantine and set `api_record_missing = 1`.
- Gap in a segment's time series ≤ 30 min → interpolate, set `interpolated = 1`. Longer → leave NULL, exclude from training.
- TomTom `confidence < 0.5` → flag `low_confidence = 1`.
- Weather unavailable → `weather_available = 0`, forward-fill ≤ 1 h only.
- No crash recorded on a segment = valid 0 (not missing).
- Data-quality report generated per pipeline run (`reports/dq_report.md`).

## Features
Lags (t-15, t-30, t-60), rolling mean/std (1 h, 3 h), sin/cos hour & weekday, peak flag, holiday flag,
weather (rain mm, temp, humidity), weather × peak, road class, lanes, length, crash count, segment historical mean.

## Modelling
- Chronological split (train → val → test by date). No random shuffling.
- Baselines: persistence (last value), logistic regression. Main: Random Forest, XGBoost. Optional: LSTM.
- Hotspots: K-Means / DBSCAN on segment × hour-of-day congestion profiles.
- Metrics: accuracy, macro-F1, per-class recall (Severe matters most), confusion matrix.
- Track experiments with MLflow.
- Explainability: SHAP on the best tree model.

## Routing
Graph from OSMnx (one-way rules applied). Edge weight = length / predicted_speed.
Edges without a sensor: use the nearest sensor's ratio on the same road class, otherwise the road-class default.
Compare congestion-aware route vs shortest-distance route: travel-time saving over N random O-D pairs.

## Repo layout
```
config/            settings.yaml (segments, thresholds, paths) — no secrets
data/raw/<source>/ data/staging/  data/curated/  data/quarantine/   (git: data/raw/ and data/quarantine/ are tracked; data/staging/* and data/curated/* are ignored except .gitkeep; data/raw/road_network/*.graphml is ignored)
src/ingestion/     one module per source
src/processing/    clean, schema-map, integrate, dq_checks
src/features/      build_features.py
src/models/        train.py, evaluate.py, clustering.py, explain.py
src/routing/       graph.py, router.py
app/               Streamlit dashboard
notebooks/         EDA only — logic lives in src/
.github/workflows/ collect_traffic.yml
reports/           dq_report.md, figures
```

## Conventions
- Python 3.11, pandas, pyarrow (Parquet for staging/curated), osmnx, networkx, scikit-learn, xgboost, shap, mlflow, streamlit, plotly.
- API keys only via `.env` / GitHub Secrets (`TOMTOM_API_KEY`). Never commit keys.
- All timestamps stored in UTC; convert to Asia/Kolkata only for features and display.
- Each pipeline step is runnable as `python -m src.<module>` and is idempotent.
- Keep functions small, type-hinted, with docstrings. Add a pytest for every ingestion parser and DQ rule.
