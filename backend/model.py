"""Training and inference model routines."""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import load_model

from .config import (
    HISTORY_PATH,
    LABELS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RESULTS_DIR,
    SCALER_PATH,
    TEST_DATA_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_DIR,
    TRAINING_META_PATH,
)
from .db import seed_students
from .features import extract_feature


_model_cache = None
_scaler_cache = None


def _ensure_dirs() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_labels() -> dict[int, str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def build_dataset(dataset_path: str) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    X: list[np.ndarray] = []
    y: list[int] = []

    students = sorted(
        [x for x in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, x))]
    )
    if not students:
        raise ValueError("No student folders found in dataset/")

    labels = {idx: name for idx, name in enumerate(students)}

    for idx, student in labels.items():
        folder = os.path.join(dataset_path, student)
        for wav in sorted(os.listdir(folder)):
            if wav.lower().endswith(".wav"):
                X.append(extract_feature(os.path.join(folder, wav)))
                y.append(idx)

    if not X:
        raise ValueError("No WAV files found in dataset/")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labels


def _assert_split_label_match(train_labels: dict[int, str], test_labels: dict[int, str]) -> None:
    train_names = [train_labels[i] for i in range(len(train_labels))]
    test_names = [test_labels[i] for i in range(len(test_labels))]
    if train_names != test_names:
        raise ValueError(
            "Train/test label mismatch. Re-download dataset to rebuild clean split. "
            f"train={train_names}, test={test_names}"
        )


def _build_model(input_dim: int, num_classes: int) -> Sequential:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model() -> dict[str, float]:
    _ensure_dirs()

    X_full_train, y_full_train, labels = build_dataset(TRAIN_DATA_DIR)
    X_test_raw, y_test, test_labels = build_dataset(TEST_DATA_DIR)
    _assert_split_label_match(labels, test_labels)

    if len(np.unique(y_full_train)) < 2:
        raise ValueError("Need at least 2 students/classes.")

    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train,
        y_full_train,
        test_size=0.15,
        random_state=42,
        stratify=y_full_train,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test_raw)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    model = _build_model(X_full_train.shape[1], len(labels))
    callbacks = [EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)]

    history = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2)

    np.savez(TEST_DATA_PATH, X_test=X_test, y_test=y_test)

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "total_train_samples": int(len(X_full_train)),
            "total_test_samples": int(len(X_test_raw)),
            "num_students": int(len(labels)),
            "students": list(labels.values()),
            "feature_dim": int(X_full_train.shape[1]),
        },
        "training": {
            "epochs_run": len(history.history.get("accuracy", [])),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
        },
        "artifacts": {
            "model_path": MODEL_PATH,
            "labels_path": LABELS_PATH,
            "scaler_path": SCALER_PATH,
        },
    }
    with open(TRAINING_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    seed_students(list(labels.values()))

    global _model_cache, _scaler_cache
    _model_cache = model
    _scaler_cache = scaler

    return {"test_accuracy": float(test_acc), "test_loss": float(test_loss)}


def _load_model_and_scaler() -> tuple[object, StandardScaler]:
    global _model_cache, _scaler_cache

    if _model_cache is None:
        _model_cache = load_model(MODEL_PATH)

    if _scaler_cache is None:
        with open(SCALER_PATH, "rb") as f:
            _scaler_cache = pickle.load(f)

    return _model_cache, _scaler_cache


def predict_from_file(file_path: str) -> tuple[int, float, np.ndarray]:
    model, scaler = _load_model_and_scaler()
    feature = extract_feature(file_path).reshape(1, -1)
    feature = scaler.transform(feature)
    probs = model.predict(feature, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    return idx, confidence, probs
