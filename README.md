# Bengaluru Traffic Congestion Prediction & Route Optimization

ML-II course project (CMR Institute of Technology). Predicts congestion level
(Low / Moderate / Severe) for South-East Bengaluru road segments 15-60 min ahead
and recommends congestion-aware routes versus plain shortest-path routes, with SHAP explanations.

See `CLAUDE.md` for the full design: data sources, integration keys, missing-data rules,
modelling and routing.

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
