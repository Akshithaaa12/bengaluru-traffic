"""MLflow tracking (local ./mlruns, experiment "metr_la_benchmark")."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow

from src.utils.config import PROJECT_ROOT

EXPERIMENT = "metr_la_benchmark"


def setup() -> None:
    # MLflow 3 puts the ./mlruns file store in maintenance mode; opt in explicitly (a sqlite URI is the alternative).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def log_run(
    name: str, params: dict[str, Any], metrics: dict[str, float], artifacts: list[Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log one run (params, metrics, artifacts such as the confusion matrix and SHAP plots); returns the run id."""
    setup()
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.set_tags(tags or {})
        for path in artifacts or []:
            mlflow.log_artifact(str(path))
        return run.info.run_id
