"""
Master Pipeline
Runs all five stages in sequence:
  1. prepare_data  — download & split IMDB
  2. train         — fit TF-IDF + LogReg, log to MLflow
  3. evaluate      — compute metrics, enforce quality gate
  4. register      — push to MLflow Model Registry
  5. serve         — launch FastAPI server (optional, --no-serve to skip)

Usage
-----
  # Full pipeline + start server
  python pipeline.py

  # Skip the server (CI / batch mode)
  python pipeline.py --no-serve

  # Tune data / model knobs
  python pipeline.py --train-size 10000 --C 0.5 --min-accuracy 0.87

  # Override thresholds
  python pipeline.py --min-accuracy 0.90 --min-f1 0.90
"""

import argparse
import logging
import subprocess
import sys
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_stage(name: str, script: str, extra_args: list[str] | None = None) -> None:
    """Run a stage script as a subprocess; raise on failure."""
    cmd = [sys.executable, script] + (extra_args or [])
    log.info("=" * 60)
    log.info("STAGE: %s", name)
    log.info("CMD  : %s", " ".join(cmd))
    log.info("=" * 60)

    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        log.error("Stage '%s' failed with exit code %d — aborting pipeline.", name, result.returncode)
        sys.exit(result.returncode)

    log.info("Stage '%s' completed successfully.\n", name)


def stage_prepare(args: argparse.Namespace) -> None:
    run_stage(
        "1 — Prepare Data",
        os.path.join(HERE, "data", "prepare_data.py"),
        [
            "--train-size", str(args.train_size),
            "--test-size",  str(args.test_size),
            "--seed",       str(args.seed),
        ],
    )


def stage_train(args: argparse.Namespace) -> None:
    run_stage(
        "2 — Train",
        os.path.join(HERE, "train", "train.py"),
        [
            "--max-features", str(args.max_features),
            "--ngram-max",    str(args.ngram_max),
            "--max-iter",     str(args.max_iter),
            "--C",            str(args.C),
            "--experiment",   args.experiment,
        ],
    )


def stage_evaluate(args: argparse.Namespace) -> None:
    run_stage(
        "3 — Evaluate",
        os.path.join(HERE, "evaluate", "evaluate.py"),
        [
            "--min-accuracy", str(args.min_accuracy),
            "--min-f1",       str(args.min_f1),
            "--experiment",   args.experiment,
        ],
    )


def stage_register(args: argparse.Namespace) -> None:
    run_stage(
        "4 — Register",
        os.path.join(HERE, "register", "register.py"),
        [
            "--model-name", args.model_name,
            "--experiment", args.experiment,
            "--stage",      args.registry_stage,
        ],
    )


def stage_serve(args: argparse.Namespace) -> None:
    """Launch the FastAPI server (blocking — stays alive until Ctrl-C)."""
    log.info("=" * 60)
    log.info("STAGE: 5 — Serve")
    log.info("Starting FastAPI on http://0.0.0.0:%d", args.port)
    log.info("Docs  → http://localhost:%d/docs", args.port)
    log.info("Press Ctrl-C to stop.")
    log.info("=" * 60)

    serve_script = os.path.join(HERE, "serve", "serve.py")

    # Use uvicorn directly for a clean startup
    cmd = [
        sys.executable, "-m", "uvicorn",
        "serve.serve:app",
        "--host", "0.0.0.0",
        "--port", str(args.port),
    ]
    try:
        subprocess.run(cmd, cwd=HERE, check=True)
    except KeyboardInterrupt:
        log.info("Server stopped by user.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full sentiment-analysis MLOps pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument("--train-size", type=int,   default=5_000)
    data_grp.add_argument("--test-size",  type=int,   default=1_000)
    data_grp.add_argument("--seed",       type=int,   default=42)

    # Training
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument("--max-features", type=int,   default=30_000)
    train_grp.add_argument("--ngram-max",    type=int,   default=2)
    train_grp.add_argument("--max-iter",     type=int,   default=500)
    train_grp.add_argument("--C",            type=float, default=1.0)

    # Evaluation
    eval_grp = parser.add_argument_group("Evaluation")
    eval_grp.add_argument("--min-accuracy", type=float, default=0.85)
    eval_grp.add_argument("--min-f1",       type=float, default=0.85)

    # Registry
    reg_grp = parser.add_argument_group("Registry")
    reg_grp.add_argument("--model-name", default="fake_news_classifier")
    reg_grp.add_argument("--registry-stage", type=str, default="Staging",
                         choices=["Staging", "Production"])

    # Serving
    serve_grp = parser.add_argument_group("Serving")
    serve_grp.add_argument("--port",     type=int,  default=8000)
    serve_grp.add_argument("--no-serve", action="store_true",
                           help="Skip stage 5 (useful in CI)")

    # Shared
    parser.add_argument("--experiment", default="fake_news_detection")

    args = parser.parse_args()

    t_start = time.time()
    log.info("Pipeline starting …")

    stage_prepare(args)
    stage_train(args)
    stage_evaluate(args)
    stage_register(args)

    elapsed = time.time() - t_start
    log.info("Pipeline stages 1-4 completed in %.1f seconds.", elapsed)

    if not args.no_serve:
        stage_serve(args)
    else:
        log.info("Skipping serve stage (--no-serve).")


if __name__ == "__main__":
    main()
