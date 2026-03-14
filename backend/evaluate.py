"""Evaluation and graph generation for trained model."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from .config import (
    EVAL_METRICS_PATH,
    HISTORY_PATH,
    LABELS_PATH,
    MODEL_PATH,
    RESULTS_DIR,
    TEST_DATA_PATH,
)


def evaluate_model() -> dict:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first.")
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError("Test data not found. Train first.")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    data = np.load(TEST_DATA_PATH)
    X_test, y_test = data["X_test"], data["y_test"]

    model = load_model(MODEL_PATH)
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test,
        y_pred,
        target_names=[labels[i] for i in range(len(labels))],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    names = [labels[i] for i in range(len(labels))]
    plt.xticks(ticks, names, rotation=45)
    plt.yticks(ticks, names)

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", color="white" if cm[i, j] > threshold else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=140)
    plt.close()

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "paths": {"confusion_matrix": cm_path},
    }
    with open(EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def plot_training_curves() -> dict[str, str]:
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError("Training history not found. Train first.")

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    epochs = list(range(1, len(train_acc) + 1))
    os.makedirs(RESULTS_DIR, exist_ok=True)

    accuracy_path = os.path.join(RESULTS_DIR, "accuracy_curve.png")
    loss_path = os.path.join(RESULTS_DIR, "loss_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=140)
    plt.close()

    return {"accuracy_curve": accuracy_path, "loss_curve": loss_path}
