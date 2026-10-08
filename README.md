# Bengaluru Traffic Congestion Prediction & Route Optimization

ML-II course project (CMR Institute of Technology). Predicts congestion level
(Low / Moderate / Severe) for South-East Bengaluru road segments 15-60 min ahead
and recommends congestion-aware routes versus plain shortest-path routes, with SHAP explanations.

See `CLAUDE.md` for the full design: data sources, integration keys, missing-data rules,
modelling and routing.

## Why METR-LA?

The Bengaluru TomTom collector is live (a GitHub Actions workflow polls 13 South-East Bengaluru segments) but has only just started
collecting, so there is not yet enough history to train a model on it. To validate the whole pipeline end to end - multi-source
ingestion, raw zones, data-quality rules, integration, curated dataset, models, SHAP, routing and the dashboard - it is first run on
**METR-LA** (Los Angeles loop-detector speeds, 4 months) plus Open-Meteo weather and US holidays. The same pipeline will be retrained
on the Bengaluru data as it accumulates. Results and limitations: `reports/benchmark/RESULTS.md`.

Benchmark commands (all run from the project root):

```bash
python -m src.benchmark.download     # raw sources + manifests inputs
python -m src.benchmark.curate       # curated parquet + DQ outputs
python -m src.benchmark.run          # models + metrics (logged to MLflow)
python -m src.benchmark.explain      # SHAP
python -m src.benchmark.ablation     # source/feature ablation
python -m src.routing.benchmark_router
python -m src.benchmark.fault_demo   # fault-injection demo
streamlit run app/streamlit_app.py   # dashboard; `mlflow ui` for experiments
```

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# create .env containing: TOMTOM_API_KEY=<your key>
```

## Layout

- `config/settings.yaml` -- study area, segments, thresholds, paths (no secrets)
- `data/raw/<source>/`, `staging/`, `curated/`, `quarantine/` -- data zones
- `src/` -- ingestion, processing, features, models, routing, utils
- `app/` -- Streamlit dashboard
- `notebooks/` -- EDA only
- `reports/` -- DQ report and figures

Each pipeline step runs as `python -m src.<module>` and is idempotent.
