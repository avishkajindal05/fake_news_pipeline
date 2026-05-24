"""
Stage 5 — Serving
FastAPI application that exposes the trained sentiment model as a REST API.

Endpoints
---------
  GET  /health          — liveness probe
  GET  /model-info      — name, version, accuracy loaded at startup
  POST /predict         — single-text prediction
  POST /predict/batch   — list of texts, returns list of predictions

Run locally
-----------
  uvicorn serve.serve:app --reload --port 8000

  or via pipeline.py which starts it programmatically.
"""

import os
import sys
import pickle
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Resolve paths regardless of where uvicorn is launched from ───────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(HERE, "..", "train")
MODEL_PATH  = os.path.join(TRAIN_DIR, "model.pkl")
REPORT_PATH = os.path.join(HERE, "..", "evaluate", "eval_report.json")

# ── Load model at startup (FastAPI lifespan) ──────────────────────────────────
_pipeline   = None
_model_meta = {}


def _load_model() -> None:
    global _pipeline, _model_meta
    if not os.path.exists(MODEL_PATH):
        log.error("model.pkl not found at %s — run pipeline stages 1-3 first.", MODEL_PATH)
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        _pipeline = pickle.load(f)

    # pull accuracy from eval report if available
    acc = "unknown"
    if os.path.exists(REPORT_PATH):
        import json
        with open(REPORT_PATH) as rf:
            report = json.load(rf)
        acc = report.get("test_accuracy", "unknown")

    _model_meta = {
        "model_path": MODEL_PATH,
        "test_accuracy": acc,
        "classes": ["fake", "real"],
    }
    log.info("Model loaded from %s  (accuracy=%s)", MODEL_PATH, acc)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detection API",
    description="fake_news_detection TF-IDF + Logistic Regression prediction classifier",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    _load_model()


# ── Schemas ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, example="This is a fake news article!")


class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, example=["This is true.", "This is false."])

class PredictionOutput(BaseModel):
    text:       str
    label:      str            # "fake" | "real"
    confidence: float          # probability of the predicted class


class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]


# ── Helpers ───────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "fake", 1: "real"}


def _predict_one(text: str) -> PredictionOutput:
    proba  = _pipeline.predict_proba([text])[0]   # shape (2,)
    cls    = int(proba.argmax())
    return PredictionOutput(
        text=text,
        label=LABEL_MAP[cls],
        confidence=round(float(proba[cls]), 4),
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.get("/model-info", tags=["ops"])
def model_info() -> dict:
    return _model_meta


@app.post("/predict", response_model=PredictionOutput, tags=["inference"])
def predict(payload: TextInput) -> PredictionOutput:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predict_one(payload.text)


@app.post("/predict/batch", response_model=BatchOutput, tags=["inference"])
def predict_batch(payload: BatchInput) -> BatchOutput:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    predictions = [_predict_one(t) for t in payload.texts]
    return BatchOutput(predictions=predictions)


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
