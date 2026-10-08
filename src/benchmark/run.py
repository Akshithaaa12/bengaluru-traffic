"""Train and evaluate the METR-LA benchmark models, write reports/benchmark/*.

Run: ``python -m src.benchmark.run``
"""
from __future__ import annotations

import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, recall_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.benchmark.prepare import CLASSES, FEATURES, SEED, build_dataset, chronological_split
from src.utils.config import project_path
from src.utils.log import get_logger

log = get_logger(__name__)
OUT = project_path("reports/benchmark")
LABELS = [0, 1, 2]


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rec = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        **{f"recall_{c.lower()}": r for c, r in zip(CLASSES, rec)},
    }


def train_models(tr: pd.DataFrame, va: pd.DataFrame) -> dict[str, tuple[object, float]]:
    """Fit the three learned models; returns {name: (model, fit_seconds)}."""
    x_tr, y_tr = tr[FEATURES], tr["y"].astype(int)
    models: dict[str, tuple[object, float]] = {}

    t0 = time.time()
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
    lr.fit(x_tr, y_tr)
    models["logistic_regression"] = (lr, time.time() - t0)
    log.info("logistic_regression fit in %.0fs", time.time() - t0)

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, min_samples_leaf=50, max_samples=0.3, random_state=SEED
    )
    rf.fit(x_tr, y_tr)
    models["random_forest"] = (rf, time.time() - t0)
    log.info("random_forest fit in %.0fs", time.time() - t0)

    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, eval_metric="mlogloss", early_stopping_rounds=20, random_state=SEED,
    )
    xgb.fit(
        x_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr),
        eval_set=[(va[FEATURES], va["y"].astype(int))], verbose=False,
    )
    models["xgboost"] = (xgb, time.time() - t0)
    log.info("xgboost fit in %.0fs (best_iteration=%s)", time.time() - t0, xgb.best_iteration)
    return models


def evaluate(models: dict[str, tuple[object, float]], splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        df = splits[split]
        y = df["y"].astype(int).to_numpy()
        rows.append({"model": "persistence", "split": split, "n": len(df), "fit_s": 0.0,
                     **score(y, df["y_now"].astype(int).to_numpy())})
        for name, (model, secs) in models.items():
            rows.append({"model": name, "split": split, "n": len(df), "fit_s": round(secs, 1),
                         **score(y, model.predict(df[FEATURES]))})
    return pd.DataFrame(rows)


def save_confusion(model: object, name: str, test: pd.DataFrame) -> None:
    y, pred = test["y"].astype(int), model.predict(test[FEATURES])
    pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=CLASSES, columns=CLASSES).to_csv(
        OUT / "confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, labels=LABELS, display_labels=CLASSES, normalize="true", values_format=".2f",
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"{name} - test set (row-normalised)\nn={len(test):,}, 30-min-ahead congestion")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def write_dq(stats: dict, splits: dict[str, pd.DataFrame]) -> None:
    pct = lambda a, b: f"{100 * a / b:.2f}%"  # noqa: E731
    s = stats
    lines = [
        "# Data-quality summary - METR-LA benchmark", "",
        f"Period: {s['start']} to {s['end']} (5-min steps, naive timestamps treated as America/Los_Angeles local time).", "",
        "## Rows per source", "",
        "| Source | Rows | Notes |", "|---|---|---|",
        f"| METR-LA speeds (all sensors) | {s['n_timestamps']:,} timestamps x 207 sensors = {s['raw_all_readings']:,} readings | mph |",
        f"| METR-LA speeds (used) | {s['n_timestamps']:,} x {s['n_sensors']} = {s['readings']:,} readings | random_state=42 |",
        f"| Open-Meteo weather | {s['weather_hours']:,} hourly values (as joined) | archive API, LA 34.05,-118.25 |",
        f"| US holidays 2012 | {s['holidays_total']} dates | `holidays` lib; {s['holiday_rows']:,} sensor-rows fall on one |", "",
        "## Missing data", "",
        "| Item | Count | % |", "|---|---|---|",
        f"| Speed == 0 (all 207 sensors) | {s['raw_all_zero']:,} | {pct(s['raw_all_zero'], s['raw_all_readings'])} |",
        f"| `sensor_missing=1` (40 sensors; speed 0 -> NaN) | {s['sensor_missing']:,} | {pct(s['sensor_missing'], s['readings'])} |",
        f"| `interpolated=1` (gap <= 30 min filled) | {s['interpolated']:,} | {pct(s['interpolated'], s['readings'])} |",
        f"| Still NaN after interpolation (gap > 30 min or at series edge; excluded) | {s['still_nan']:,} | {pct(s['still_nan'], s['readings'])} |",
        f"| `weather_available=0` timestamps | {s['weather_unavailable_rows']:,} | {pct(s['weather_unavailable_rows'], s['n_timestamps'])} |",
        *[f"| Weather `{k}` still NaN after <=1 h ffill | - | {v:.2f}% |" for k, v in s["weather_missing_pct"].items()], "",
        "Speed 0 is treated as a sensor failure, never as a real zero. Rows with any NaN feature or target "
        "(lags/rolling windows touching a long gap) are dropped before modelling.", "",
        "## Modelling table", "",
        f"- Rows before dropping: {s['rows_total']:,}; kept: {s['rows_complete']:,} ({pct(s['rows_complete'], s['rows_total'])})", "",
        "| Split | Rows | Low | Moderate | Severe |", "|---|---|---|---|---|",
    ]
    for name, df in splits.items():
        share = df["y"].astype(int).value_counts(normalize=True).reindex(LABELS, fill_value=0)
        lines.append(f"| {name} | {len(df):,} | " + " | ".join(f"{100 * share[i]:.1f}%" for i in LABELS) + " |")
    (OUT / "dq_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    df, stats = build_dataset()
    splits = chronological_split(df, stats["n_timestamps"])
    log.info("rows: %s", {k: len(v) for k, v in splits.items()})
    write_dq(stats, splits)

    models = train_models(splits["train"], splits["val"])
    metrics = evaluate(models, splits)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    val = metrics[(metrics.split == "val") & metrics.model.isin(models)]
    best = val.loc[val.macro_f1.idxmax(), "model"]       # chosen on validation, not test
    model = models[best][0]
    save_confusion(model, best, splits["test"])
    joblib.dump({"model": model, "features": FEATURES, "classes": ["Low", "Moderate", "Severe"]},
                OUT / "models" / "best_model.pkl", compress=3)

    print(f"\nBest model (by validation macro-F1): {best}\n")
    for split in ("test", "val"):
        print(f"== {split} ==")
        print(metrics[metrics.split == split].drop(columns="split").to_string(index=False, float_format="%.3f"))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
