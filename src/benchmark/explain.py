"""SHAP explanations for the final (XGBoost) model on a 2,000-row test sample.

Writes shap_summary.png (beeswarm, Severe class), shap_bar.png and shap_top_features.csv.

Run: ``python -m src.benchmark.explain``
"""
from __future__ import annotations

import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.benchmark.prepare import CLASSES, FEATURES, SEED, build_dataset, chronological_split
from src.utils.config import project_path
from src.utils.log import get_logger

log = get_logger(__name__)
OUT = project_path("reports/benchmark")
SAMPLE = 2000
SEVERE = CLASSES.index("Severe")


def class_shap(values: np.ndarray | list, cls: int) -> np.ndarray:
    """SHAP values for one class from either (n, f, c) arrays or a per-class list."""
    return values[cls] if isinstance(values, list) else values[:, :, cls]


def main() -> int:
    model = joblib.load(OUT / "models" / "best_model.pkl")["model"]
    df, stats = build_dataset()
    test = chronological_split(df, stats["n_timestamps"])["test"]
    x = test[FEATURES].sample(n=SAMPLE, random_state=SEED)

    values = shap.TreeExplainer(model).shap_values(x)
    sv = class_shap(values, SEVERE)
    all_classes = np.mean([np.abs(class_shap(values, c)).mean(axis=0) for c in range(len(CLASSES))], axis=0)

    top = pd.DataFrame({
        "feature": FEATURES, "mean_abs_shap_severe": np.abs(sv).mean(axis=0), "mean_abs_shap_all_classes": all_classes,
    }).sort_values("mean_abs_shap_severe", ascending=False)
    top["share_severe"] = top["mean_abs_shap_severe"] / top["mean_abs_shap_severe"].sum()
    top.to_csv(OUT / "shap_top_features.csv", index=False)

    plt.figure()
    shap.summary_plot(sv, x, show=False, max_display=15)
    plt.title("SHAP - Severe class (XGBoost, 2,000 test rows)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(sv, x, plot_type="bar", show=False, max_display=15)
    plt.title("Mean |SHAP| - Severe class")
    plt.tight_layout()
    plt.savefig(OUT / "shap_bar.png", dpi=150)
    plt.close()
    log.info("SHAP done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
