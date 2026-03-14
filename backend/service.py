"""High-level orchestration for attendance operations."""

from __future__ import annotations

import os

from .audio import record_temp_wav
from .config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DURATION, DEFAULT_SAMPLE_RATE
from .db import init_db, mark_attendance
from .model import load_labels, predict_from_file


def predict_and_optionally_mark(
    duration: float = DEFAULT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    expected_name: str | None = None,
) -> dict:
    temp_wav = record_temp_wav(duration=duration, sample_rate=sample_rate)
    try:
        return predict_file_and_optionally_mark(
            file_path=temp_wav,
            threshold=threshold,
            expected_name=expected_name,
            source="microphone",
        )
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


def predict_file_and_optionally_mark(
    file_path: str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    expected_name: str | None = None,
    source: str = "dataset_test",
) -> dict:
    init_db()

    labels = load_labels()
    idx, confidence, probs = predict_from_file(file_path)
    name = labels[idx]

    marked = False
    block_reason = ""

    if confidence < threshold:
        block_reason = "low_confidence"
    elif expected_name and name != expected_name:
        block_reason = "expected_name_mismatch"
    else:
        mark_attendance(name, confidence, source=source)
        marked = True

    all_probs = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return {
        "name": name,
        "confidence": confidence,
        "all_probs": all_probs,
        "attendance_marked": marked,
        "threshold": threshold,
        "expected_name": expected_name,
        "block_reason": block_reason,
        "source": source,
    }
