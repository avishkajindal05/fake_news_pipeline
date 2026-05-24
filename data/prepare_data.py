"""
Stage 1 — Data Preparation
Downloads the fake news dataset, sub-samples it for speed, and writes
train.csv / test.csv to  fake_news_pipeline/data/
"""

import os
import argparse
import logging
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))


def prepare(train_size: int = 5_000, test_size: int = 1_000, seed: int = 42) -> None:
    log.info("Loading news data dataset from HuggingFace …")
    dataset = load_dataset("GonzaloA/fake_news")

    train_df = (
        dataset["train"]
        .to_pandas()
        .sample(n=min(train_size, len(dataset["train"])), random_state=seed)
        .reset_index(drop=True)
    )

    test_df = (
        dataset["test"]
        .to_pandas()
        .sample(n=min(test_size, len(dataset["test"])), random_state=seed)
        .reset_index(drop=True)
    )

    # Rename HuggingFace column to something explicit
    train_df = train_df.rename(columns={"label": "is_real"})
    test_df  = test_df.rename(columns={"label": "is_real"})

    # 0 → "fake", 1 → "real"  (keep numeric too for sklearn)
    train_df["sentiment_label"] = train_df["is_real"].map({0: "fake", 1: "real"})
    test_df["sentiment_label"]  = test_df["is_real"].map({0: "fake", 1: "real"})
    train_path = os.path.join(HERE, "train.csv")
    test_path  = os.path.join(HERE, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    log.info("Saved %d training rows  → %s", len(train_df), train_path)
    log.info("Saved %d test rows      → %s", len(test_df),  test_path)
    log.info("Class distribution (train):\n%s", train_df["sentiment_label"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fake news data")
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--test-size",  type=int, default=1_000)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    prepare(train_size=args.train_size, test_size=args.test_size, seed=args.seed)
