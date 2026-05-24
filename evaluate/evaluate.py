"""
Stage 3 — Evaluation & Quality Gate
Loads the trained model and the held-out test set, computes a full suite of
metrics, logs them to the active (or most-recent) MLflow run, then enforces a
quality gate.  The script exits with code 1 if the gate fails so the pipeline
can short-circuit before registering or deploying a poor model.

Quality gate (defaults)
-----------------------
  accuracy  >= 0.85
  f1_score  >= 0.85
"""

import os
import sys
import argparse
import logging
import pickle
import json

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(HERE, "..", "data")
TRAIN_DIR  = os.path.join(HERE, "..", "train")
MODEL_PATH = os.path.join(TRAIN_DIR, "model.pkl")
REPORT_PATH = os.path.join(HERE, "eval_report.json")


def evaluate(
    min_accuracy: float = 0.85,
    min_f1:       float = 0.85,
    experiment:   str   = "sentiment_analysis",
) -> dict:
    # ── Load artefacts ────────────────────────────────────────────────────────
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found at {test_path}. Run Stage 1 first.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}. Run Stage 2 first.")

    log.info("Loading test data and model …")
    df = pd.read_csv(test_path)
    X_test = df["text"].astype(str).tolist()
    y_test = df["is_real"].tolist()

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    # ── Predictions ───────────────────────────────────────────────────────────
    log.info("Running inference on %d test samples …", len(X_test))
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = {
        "test_accuracy":  round(accuracy_score(y_test, y_pred),              4),
        "test_f1":        round(f1_score(y_test, y_pred),                    4),
        "test_precision": round(precision_score(y_test, y_pred),             4),
        "test_recall":    round(recall_score(y_test, y_pred),                4),
        "test_roc_auc":   round(roc_auc_score(y_test, y_pred_prob),          4),
    }

    cm = confusion_matrix(y_test, y_pred).tolist()
    report_str = classification_report(
        y_test, y_pred, target_names=["fake", "real"]
    )

    log.info("\n%s", report_str)
    log.info("Confusion matrix:\n  TN=%d  FP=%d\n  FN=%d  TP=%d",
             cm[0][0], cm[0][1], cm[1][0], cm[1][1])

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    mlflow.set_experiment(experiment)
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time DESC"])
    if runs.empty:
        log.warning("No MLflow runs found — metrics will not be logged to MLflow.")
    else:
        run_id = runs.iloc[0]["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.log_dict({"confusion_matrix": cm}, "confusion_matrix.json")
            log.info("Metrics logged to MLflow run %s", run_id)

    # ── Write local report ────────────────────────────────────────────────────
    report = {**metrics, "confusion_matrix": cm, "classification_report": report_str}
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Evaluation report saved → %s", REPORT_PATH)

    # ── Quality gate ──────────────────────────────────────────────────────────
    passed = True
    if metrics["test_accuracy"] < min_accuracy:
        log.error(
            "QUALITY GATE FAILED: accuracy %.4f < threshold %.4f",
            metrics["test_accuracy"], min_accuracy,
        )
        passed = False
    if metrics["test_f1"] < min_f1:
        log.error(
            "QUALITY GATE FAILED: F1 %.4f < threshold %.4f",
            metrics["test_f1"], min_f1,
        )
        passed = False

    if passed:
        log.info("Quality gate PASSED (accuracy=%.4f, f1=%.4f)", metrics["test_accuracy"], metrics["test_f1"])
    else:
        log.error("Quality gate FAILED — aborting pipeline.")
        sys.exit(1)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sentiment model")
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-f1",       type=float, default=0.85)
    parser.add_argument("--experiment",   type=str,   default="sentiment_analysis")
    args = parser.parse_args()

    evaluate(
        min_accuracy=args.min_accuracy,
        min_f1=args.min_f1,
        experiment=args.experiment,
    )
