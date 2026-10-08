"""Assemble reports/benchmark/RESULTS.md from the saved outputs (numbers stay in sync).

Run: ``python -m src.benchmark.results_md``
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.utils.config import project_path

BENCH = project_path("reports/benchmark")
LIVE = project_path("data/raw/traffic_api")


def _cell(v: object, fmt: str) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(_cell(v, fmt) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, *body])


def section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def main() -> int:
    metrics = pd.read_csv(BENCH / "metrics.csv")
    test = metrics[metrics["split"] == "test"].drop(columns=["split", "fit_s"])
    shap_df = pd.read_csv(BENCH / "shap_top_features.csv")
    top10 = shap_df.head(10)[["feature", "mean_abs_shap_severe", "share_severe"]].copy()
    top10["share_severe"] = (top10["share_severe"] * 100).round(1).astype(str) + "%"
    top10["mean_abs_shap_severe"] = top10["mean_abs_shap_severe"].astype(float)
    route = json.loads((BENCH / "routing_meta.json").read_text())
    n_live = len(list(LIVE.glob("*/run_*.json*")))
    sources = pd.read_csv(BENCH / "sources.csv")[["source", "role", "format", "ingestion", "raw_zone", "integration_key", "why_useful"]]
    sources.loc[sources["source"].str.startswith("Bengaluru"), "why_useful"] += f" ({n_live} raw run files so far)"
    rules = pd.read_csv(BENCH / "dq_rules.csv")[["rule_id", "source", "field", "requirement", "detection", "handling"]]
    missing = pd.read_csv(BENCH / "missing_by_source.csv")

    abl = pd.read_csv(BENCH / "ablation.csv").set_index("variant")

    def abl_line(v: str) -> str:
        r = abl.loc[v]
        return f"macro-F1 {r.test_macro_f1:.3f} ({r.delta_macro_f1:+.3f}), Severe recall {r.test_recall_severe:.3f} ({r.delta_recall_severe:+.3f})"

    def verdict(v: str) -> str:
        return "no measurable contribution" if abl.loc[v, "delta_macro_f1"] > -0.005 else "a real contribution"

    abl_table = abl.reset_index()[["variant", "n_features", "test_accuracy", "test_macro_f1", "test_recall_severe",
                                   "delta_macro_f1", "delta_recall_severe"]]
    xgb = test[test["model"] == "xgboost"].iloc[0]
    pers = test[test["model"] == "persistence"].iloc[0]
    text = f"""# METR-LA benchmark - results

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

{md_table(sources)}

### Schema differences

The sources share nothing except time: METR-LA is a wide HDF5 matrix (timestamp x 207 sensor columns) plus CSVs, Open-Meteo is
column-oriented JSON arrays with ISO time strings, and the holiday list is a two-column CSV. They have different granularity
(5 min, 1 h, 1 day), so each joins on a different key (see [`source_schemas.md`](source_schemas.md)).

## Data-quality rules

Mandatory fields (`sensor_id`, `timestamp`, `speed`) must be present or the record/sensor is rejected or excluded; optional
sources (weather, holiday) only enrich a row and are flagged when missing (`weather_available`).

{md_table(rules)}

## Missing data by source

{md_table(missing)}

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

{md_table(test)}

XGBoost reaches Severe recall {xgb['recall_severe']:.3f} versus {pers['recall_severe']:.3f} for persistence, but with lower
accuracy ({xgb['accuracy']:.3f} vs {pers['accuracy']:.3f}) because class weighting trades Low-class recall for Moderate/Severe
recall. Confusion matrix: `confusion_matrix.png`. Validation metrics are in `metrics.csv`.

## SHAP - top features for the Severe class

TreeExplainer on XGBoost, 2,000 random test rows (`shap_summary.png`, `shap_bar.png`, `shap_top_features.csv`).

{md_table(top10, "{:.3f}")}

## What each source contributes (ablation)

XGBoost retrained with feature groups removed (same splits and hyperparameters; the final model is unchanged). Test set:

{md_table(abl_table, "{:.4f}")}

- **Weather** (temperature, precipitation, humidity): without it {abl_line("no_weather")} - {verdict("no_weather")}. It is a single
  Los Angeles point shared by all 40 sensors, and rain is rare in this period.
- **Holidays**: without them {abl_line("no_holiday")} - {verdict("no_holiday")}. The 119-day range contains one holiday (Memorial Day).
- **Both optional sources removed**: {abl_line("no_weather_no_holiday")}.
- **`free_flow_mph`**: without it {abl_line("no_free_flow_mph")} - {verdict("no_free_flow_mph")}; it carries per-sensor information (it acts
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

Directed graph of {route['nodes']} sensors / {route['edges']} edges (road distance < 5 km, largest weakly connected component of
the 40 sampled sensors). Edge time = distance / speed at the destination sensor. For {route['pairs']} random O-D pairs at the
weekday 08:00 test timestamp {route['forecast_origin']}, the congestion-aware route is chosen with XGBoost-predicted speeds
and both routes are timed with the **actual** speeds at {route['evaluated_at']}.

- **Mean saving: {route['mean_saving_pct']:.2f}%, median saving: {route['median_saving_pct']:.2f}%**
- The two routes differ in {route['pairs_with_different_routes']} of {route['pairs']} pairs (mean saving {route['mean_saving_pct_differing_pairs']}% over those);
  {route['pairs_improved']} pairs improved, {route['pairs_worse']} got worse.

Output: `routing_results.csv`, `routing_example.png` (the example shown is the best of the {route['pairs']} pairs, not a typical one).

## Limitations

- **Not Bengaluru.** Models are trained and tested on Los Angeles 2012 data; Bengaluru collection only started on 2026-10-07.
- **XGBoost is not clearly better than persistence overall.** Accuracy is lower ({xgb['accuracy']:.3f} vs {pers['accuracy']:.3f}) and macro-F1
  only marginally higher ({xgb['macro_f1']:.3f} vs {pers['macro_f1']:.3f}); its advantage is Severe/Moderate recall, bought with more false
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
"""
    (BENCH / "RESULTS.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
