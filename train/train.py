"""
Stage 2 — Model Training
Trains a TF-IDF + Logistic Regression pipeline on the prepared fake news data.
Every parameter and metric is logged to MLflow so runs are fully reproducible.

Artifacts written
-----------------
  train/model.pkl          — serialised sklearn Pipeline
  train/vectorizer.pkl     — the fitted TfidfVectorizer (standalone, for serve)
"""

import os
import argparse
import logging
import pickle
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(HERE, "..", "data")
MODEL_PATH  = os.path.join(HERE, "model.pkl")


def train(
    max_features: int = 30_000,
    ngram_max:    int = 2,
    max_iter:     int = 500,
    C:            float = 1.0,
    experiment:   str = "sentiment_analysis",
) -> str:
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"train.csv not found at {train_path}. Run Stage 1 (prepare_data.py) first."
        )

    log.info("Loading training data …")
    df = pd.read_csv(train_path)
    X_train = df["text"].astype(str).tolist()
    y_train = df["is_real"].tolist()          # 0 / 1

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="tfidf_logreg") as run:
        run_id = run.info.run_id
        log.info("MLflow run id: %s", run_id)

        # ── Parameters ──────────────────────────────────────────────────────
        params = {
            "max_features": max_features,
            "ngram_range":  f"(1,{ngram_max})",
            "C":            C,
            "max_iter":     max_iter,
            "solver":       "lbfgs",
        }
        mlflow.log_params(params)

        # ── Build pipeline ───────────────────────────────────────────────────
        pipeline = Pipeline([
            ("tfidf_word", TfidfVectorizer(
                max_features=20_000,
                ngram_range=(1, 3),       # trigrams instead of bigrams
                min_df=2,                  # NEW: filter rare terms
                sublinear_tf=True,
                analyzer="word",
            )),
            ("clf", LogisticRegression(C=0.3, max_iter=500, solver="saga")),  # saga solver
        ])

        log.info("Training … (n=%d samples)", len(X_train))
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        # ── Training-set metrics (quick sanity check) ────────────────────────
        y_pred = pipeline.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred)
        train_f1  = f1_score(y_train, y_pred)

        mlflow.log_metrics({
            "train_accuracy": round(train_acc, 4),
            "train_f1":       round(train_f1,  4),
            "train_time_sec": round(elapsed,   2),
        })

        log.info("Train accuracy: %.4f | F1: %.4f | Time: %.1fs", train_acc, train_f1, elapsed)

        # ── Persist model ────────────────────────────────────────────────────
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(pipeline, f)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_artifact(MODEL_PATH, artifact_path="model_pickle")

        log.info("Model saved → %s", MODEL_PATH)

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model")
    parser.add_argument("--max-features", type=int,   default=30_000)
    parser.add_argument("--ngram-max",    type=int,   default=2)
    parser.add_argument("--max-iter",     type=int,   default=500)
    parser.add_argument("--C",            type=float, default=1.0)
    parser.add_argument("--experiment",   type=str,   default="sentiment_analysis")
    args = parser.parse_args()

    train(
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        max_iter=args.max_iter,
        C=args.C,
        experiment=args.experiment,
    )
