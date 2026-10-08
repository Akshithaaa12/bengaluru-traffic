"""Assemble reports/benchmark/RESULTS.md from the saved outputs (numbers stay in sync).

Run: ``python -m src.benchmark.results_md``
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from src.utils.config import project_path

BENCH = project_path("reports/benchmark")
LIVE = project_path("data/raw/traffic_api")


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
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
    dq = (BENCH / "dq_summary.md").read_text(encoding="utf-8")
    n_live = len(list(LIVE.glob("*/run_*.json*")))

    xgb = test[test["model"] == "xgboost"].iloc[0]
    pers = test[test["model"] == "persistence"].iloc[0]
    text = f"""# METR-LA benchmark - results

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
| Bengaluru TomTom live collector | `data/raw/traffic_api/` | Flow Segment API, 13 validated South-East Bengaluru segments, every 15 min via GitHub Actions; {n_live} raw run files at the time of writing. **Not used for modelling yet** (too little history). |

## Missing-data handling

Speed 0 is a sensor failure, not a real zero: it is set to NaN and flagged `sensor_missing=1`. Gaps of 30 minutes or
less are linearly interpolated (`interpolated=1`); longer gaps stay NaN and those rows are excluded from training and
evaluation. Weather is forward-filled for at most 1 hour.

{section(dq, "## Missing data", "## Modelling table")}

Full detail: [`dq_summary.md`](dq_summary.md).

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
- **Time zone.** Naive METR-LA timestamps are treated as America/Los_Angeles local time; daylight-saving handling is unverified.
- **No unit tests yet** for the benchmark modules.
"""
    (BENCH / "RESULTS.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
