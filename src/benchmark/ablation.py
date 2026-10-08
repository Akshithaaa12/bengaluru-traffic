"""Ablation: how much does each source / feature group add? Retrains XGBoost with features removed.

The final model is NOT replaced. Writes reports/benchmark/ablation.csv and logs every variant to MLflow.

Run: ``python -m src.benchmark.ablation``
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.benchmark.explain import make_shap_plots
from src.benchmark.prepare import FEATURES, SEED, build_dataset, chronological_split
from src.benchmark.run import confusion_png, fit_xgb, score
from src.benchmark.tracking import log_run
from src.utils.config import project_path
from src.utils.log import get_logger

log = get_logger(__name__)
OUT = project_path("reports/benchmark")
WEATHER = ["temp_c", "precip_mm", "humidity"]
VARIANTS: dict[str, list[str]] = {
    "all_features": [],
    "no_weather": WEATHER,
    "no_holiday": ["is_holiday"],
    "no_free_flow_mph": ["free_flow_mph"],
    "no_weather_no_holiday": WEATHER + ["is_holiday"],   # extra: both optional sources removed
}


def run_ablation(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, log_mlflow: bool = True) -> pd.DataFrame:
    rows = []
    y_test = test["y"].astype(int).to_numpy()
    sample_idx = test.sample(n=2000, random_state=SEED).index
    for variant, removed in VARIANTS.items():
        feats = [f for f in FEATURES if f not in removed]
        model = fit_xgb(train, val, feats)
        pred = model.predict(test[feats])
        test_s, val_s = score(y_test, pred), score(val["y"].astype(int).to_numpy(), model.predict(val[feats]))
        rows.append({
            "variant": variant, "removed_features": ",".join(removed) or "-", "n_features": len(feats),
            "best_iteration": int(model.best_iteration),
            **{f"test_{k}": v for k, v in test_s.items()}, "val_macro_f1": val_s["macro_f1"],
        })
        if log_mlflow:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                confusion_png(y_test, pred, f"xgboost {variant}", tmp_dir / "confusion_matrix.png")
                make_shap_plots(model, test.loc[sample_idx, feats], tmp_dir, f"XGBoost {variant}")
                log_run(
                    f"ablation_{variant}",
                    {"variant": variant, "removed_features": ",".join(removed) or "-", "n_features": len(feats),
                     "features": ",".join(feats), "model": "xgboost", "class_weight": "balanced"},
                    {**{f"test_{k}": v for k, v in test_s.items()}, "val_macro_f1": val_s["macro_f1"]},
                    [tmp_dir / "confusion_matrix.png", tmp_dir / "shap_summary.png", tmp_dir / "shap_bar.png"],
                    {"kind": "ablation"},
                )
        log.info("%s: test macro-F1 %.3f, Severe recall %.3f", variant, test_s["macro_f1"], test_s["recall_severe"])
    df = pd.DataFrame(rows)
    base = df.loc[df["variant"] == "all_features"].iloc[0]
    df["delta_macro_f1"] = df["test_macro_f1"] - base["test_macro_f1"]
    df["delta_recall_severe"] = df["test_recall_severe"] - base["test_recall_severe"]
    return df


def main() -> int:
    df, stats = build_dataset()
    splits = chronological_split(df, stats["n_timestamps"])
    result = run_ablation(splits["train"], splits["val"], splits["test"])
    result.to_csv(OUT / "ablation.csv", index=False)
    cols = ["variant", "n_features", "test_accuracy", "test_macro_f1", "test_recall_severe", "delta_macro_f1", "delta_recall_severe"]
    print(result[cols].to_string(index=False, float_format="%.4f"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
