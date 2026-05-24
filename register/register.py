"""
Stage 4 — Model Registration
Finds the most-recent MLflow run for the experiment, registers the model in the
MLflow Model Registry, and transitions it to the "Staging" stage.

If a model already exists in the registry, the new version is added and the
previous "Staging" version is archived automatically.
"""

import os
import argparse
import logging
import time

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentiment_classifier"
DEFAULT_EXPERIMENT  = "sentiment_analysis"


def register(
    model_name: str = DEFAULT_MODEL_NAME,
    experiment:  str = DEFAULT_EXPERIMENT,
    stage:       str = "Staging",
) -> str:
    """Register the best run's model and return the new version string."""
    mlflow.set_experiment(experiment)
    client = MlflowClient()

    # ── Find the most-recent run ──────────────────────────────────────────────
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"No runs found in experiment '{experiment}'. Run Stages 2 & 3 first."
        )

    run_id     = runs.iloc[0]["run_id"]
    run_acc    = runs.iloc[0].get("metrics.test_accuracy", "N/A")
    run_f1     = runs.iloc[0].get("metrics.test_f1",       "N/A")
    model_uri  = f"runs:/{run_id}/model"

    log.info("Registering run %s  (accuracy=%s, f1=%s)", run_id, run_acc, run_f1)

    # ── Register ──────────────────────────────────────────────────────────────
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = result.version
    log.info("Registered as  '%s'  version %s", model_name, version)

    # MLflow registration is async; wait until it's ready
    for _ in range(30):
        mv = client.get_model_version(name=model_name, version=version)
        if mv.status == "READY":
            break
        log.info("  … waiting for model version to become READY (status=%s)", mv.status)
        time.sleep(2)

    # ── Archive previous Staging versions ────────────────────────────────────
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == stage and mv.version != version:
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived"
            )
            log.info("Archived previous %s version %s", stage, mv.version)

    # ── Transition new version to target stage ────────────────────────────────
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=False,
    )
    log.info("Model '%s' v%s transitioned to stage '%s'", model_name, version, stage)

    # ── Annotate with run metadata ────────────────────────────────────────────
    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            f"TF-IDF + LogisticRegression  |  run_id={run_id}  |  "
            f"accuracy={run_acc}  f1={run_f1}"
        ),
    )

    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register sentiment model in MLflow")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--stage",      type=str, default="Staging",
                        choices=["Staging", "Production"])
    args = parser.parse_args()

    register(
        model_name=args.model_name,
        experiment=args.experiment,
        stage=args.stage,
    )
