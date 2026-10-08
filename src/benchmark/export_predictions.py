"""Export test-set predictions and feature importances from the saved best model (no retraining).

Writes ``reports/benchmark/test_predictions.parquet`` and ``feature_importance.csv`` so the
Streamlit app only reads saved outputs.

Run: ``python -m src.benchmark.export_predictions``
"""
from __future__ import annotations

import sys

import joblib
import pandas as pd

from src.benchmark.prepare import FEATURES, build_dataset, chronological_split, load_raw_speeds, pick_sensors
from src.utils.config import project_path
from src.utils.log import get_logger

log = get_logger(__name__)
OUT = project_path("reports/benchmark")
LOCATIONS = project_path("data/raw/metr_la/sensor_locations_la.csv")


def final_estimator(model: object) -> object:
    """Unwrap a sklearn Pipeline to its last step."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


def main() -> int:
    bundle = joblib.load(OUT / "models" / "best_model.pkl")
    model = bundle["model"]

    df, stats = build_dataset()
    test = chronological_split(df, stats["n_timestamps"])["test"].copy()

    sensor_ids = pick_sensors(load_raw_speeds()).columns.astype(str).to_numpy()
    loc = pd.read_csv(LOCATIONS, dtype={"sensor_id": str}).set_index("sensor_id")
    ids = sensor_ids[test["sensor"].to_numpy()]
    out = pd.DataFrame({
        "time": test["time"].to_numpy(), "sensor_id": ids,
        "lat": loc.loc[ids, "latitude"].to_numpy(), "lon": loc.loc[ids, "longitude"].to_numpy(),
        "y_true": test["y"].astype(int).to_numpy(),
        "y_pred": model.predict(test[FEATURES]).astype(int),
    })
    out.to_parquet(OUT / "test_predictions.parquet", index=False)

    est = final_estimator(model)
    if hasattr(est, "feature_importances_"):
        pd.DataFrame({"feature": FEATURES, "importance": est.feature_importances_}).sort_values(
            "importance", ascending=False
        ).to_csv(OUT / "feature_importance.csv", index=False)
    log.info("exported %d test predictions for %d sensors", len(out), out["sensor_id"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
