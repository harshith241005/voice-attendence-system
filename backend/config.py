"""Shared paths and constants."""

from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DATA_DIR = os.path.join(DATASET_DIR, "train")
TEST_DATA_DIR = os.path.join(DATASET_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")

DB_PATH = os.path.join(DATABASE_DIR, "attendance.db")
MODEL_PATH = os.path.join(MODELS_DIR, "voice_model.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

TEST_DATA_PATH = os.path.join(RESULTS_DIR, "test_data.npz")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.json")
EVAL_METRICS_PATH = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
TRAINING_META_PATH = os.path.join(RESULTS_DIR, "training_metadata.json")

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION = 3.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
