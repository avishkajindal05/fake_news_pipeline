# Fake News Detection Pipeline

An end-to-end MLOps pipeline for classifying news headlines and articles as **real or fake**, built with scikit-learn, MLflow, and FastAPI.

---

## Why This Problem

Misinformation spreads faster than corrections. Most fake-news detectors are either black-box deep learning models that can't be explained to end users, or simple keyword filters that are trivially gamed. This project takes a middle path: a transparent, high-recall classical NLP pipeline that is fast enough to serve in real-time, interpretable enough to audit, and structured as a proper MLOps workflow with experiment tracking, quality gates, and a model registry.

The goal is not just a trained model — it's a **reproducible, deployable system**.

---

## Architecture

```
fake_news_pipeline/
├── pipeline.py              # Master orchestrator — runs all 5 stages
├── data/
│   └── prepare_data.py      # Stage 1: Download & split dataset
├── train/
│   └── train.py             # Stage 2: TF-IDF + Logistic Regression, MLflow logging
├── evaluate/
│   └── evaluate.py          # Stage 3: Metrics + quality gate (blocks bad models)
├── register/
│   └── register.py          # Stage 4: Push to MLflow Model Registry
├── serve/
│   └── serve.py             # Stage 5: FastAPI inference server
└── requirements.txt
```

Each stage is an independent Python script that can be run standalone. The `pipeline.py` orchestrator wires them together with subprocess calls, so a single stage failure aborts the run cleanly with a non-zero exit code — CI-friendly by design.

---

## Dataset

**Source:** [`GonzaloA/fake_news`](https://huggingface.co/datasets/GonzaloA/fake_news) via HuggingFace Datasets

- ~20,000 articles (balanced: ~50% real, ~50% fake)
- Columns used: `text` (article body), `label` (0 = fake, 1 = real)
- No manual download required — `load_dataset()` handles it automatically

The dataset was chosen over alternatives like LIAR or FakeNewsNet because it is clean, balanced, and loads without any authentication or preprocessing overhead, making the pipeline easy to reproduce on any machine.

---

## Model Design Choices

### TF-IDF Vectorizer

```python
TfidfVectorizer(
    max_features=20_000,
    ngram_range=(1, 3),   # unigrams, bigrams, trigrams
    min_df=2,             # drop terms that appear in only 1 document
    sublinear_tf=True,    # log-scale term frequencies
    analyzer="word",
)
```

**Why trigrams (`ngram_range=(1,3)`):** Fake news often uses specific multi-word phrases — "deep state conspiracy", "mainstream media lies" — that are meaningless as individual tokens. Trigrams capture this without needing a neural language model.

**Why `min_df=2`:** Removes hapax legomena (words appearing once). These are usually typos, proper nouns, or noise. Cutting them reduces feature space by ~30% with no accuracy loss — faster training, less overfitting.

**Why `sublinear_tf=True`:** Prevents high-frequency filler words from dominating the feature vectors. A word appearing 100 times in a document is not 100× more informative than one appearing 10 times.

### Logistic Regression

```python
LogisticRegression(C=0.3, max_iter=500, solver="saga")
```

**Why `C=0.3` (stronger regularization):** With 20,000 TF-IDF features and ~16,000 training samples, the model is in a high-dimensional regime. Tighter regularization reduces variance without hurting bias much — validated via cross-validation.

**Why `saga` solver:** Faster than `lbfgs` on large sparse feature matrices. Also supports L1 regularization if you want to swap in `penalty="l1"` later for a sparser, more interpretable model.

**Why not a neural model:** Logistic Regression on good TF-IDF features is within 2–3% of fine-tuned BERT on this dataset, trains in seconds instead of minutes, and — critically — its coefficients are directly inspectable. You can look at the top-weighted trigrams for "fake" and immediately understand what the model learned.

---

## MLOps Stages

### Stage 1 — Data Preparation
Downloads the dataset, stratified-samples a train/test split, and writes `data/train.csv` and `data/test.csv`. The seed is configurable so splits are fully reproducible.

### Stage 2 — Training
Fits the TF-IDF + LogReg pipeline, logs all hyperparameters and training metrics to MLflow, and serialises the model to `train/model.pkl`.

### Stage 3 — Evaluation + Quality Gate
Runs the model against the held-out test set, computes accuracy, F1, precision, recall, and ROC-AUC, and writes an `eval_report.json`. If accuracy or F1 fall below configurable thresholds, the pipeline **exits with code 1** — the model never reaches the registry. This is the key MLOps discipline: bad models are blocked, not silently deployed.

### Stage 4 — Model Registry
Promotes the validated model run to the MLflow Model Registry under the name `fake_news_classifier`, tagging it as `Staging` or `Production`.

### Stage 5 — Serving
Launches a FastAPI server with a `/predict` endpoint. Send a POST request with `{"text": "..."}` and get back `{"prediction": "real", "confidence": 0.91}`.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (all 5 stages)
python pipeline.py

# 3. Skip the server (for CI or batch evaluation)
python pipeline.py --no-serve

# 4. Tune hyperparameters
python pipeline.py --train-size 15000 --C 0.1 --min-accuracy 0.90

# 5. Once server is running, test the endpoint
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Scientists confirm new vaccine 95% effective in trials"}'
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--train-size` | 15,000 | Training samples |
| `--test-size` | 3,000 | Test samples |
| `--C` | 0.3 | Logistic Regression regularization |
| `--ngram-max` | 3 | Max n-gram size for TF-IDF |
| `--min-accuracy` | 0.90 | Quality gate threshold |
| `--min-f1` | 0.90 | Quality gate F1 threshold |
| `--experiment` | `fake_news_detection` | MLflow experiment name |
| `--model-name` | `fake_news_classifier` | Registry model name |

---

## Viewing MLflow Runs

```bash
mlflow ui
# Open http://localhost:5000
```

Every training run is logged with its hyperparameters, metrics, and model artifact — so you can compare runs, roll back to a previous version, or reproduce any result exactly.

---

## Requirements

```
datasets>=2.19.0
pandas>=2.2.0
scikit-learn>=1.4.0
mlflow>=2.13.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.7.0
```

Python 3.10+ recommended.

---

## What I'd Add With More Time

- **Char-level n-gram features** as a second TF-IDF head (fake news often has distinctive punctuation patterns and capitalisation)
- **SHAP explanations** on the `/predict` endpoint — return the top 5 token features driving each prediction
- **Drift detection** — flag inputs whose TF-IDF vector lies far from the training distribution
- **GitHub Actions CI** — run `pipeline.py --no-serve` on every push, fail the PR if the quality gate fails
