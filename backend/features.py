"""Audio feature extraction for model input."""

from __future__ import annotations

import numpy as np
import librosa


def extract_feature(
    file_path: str,
    sample_rate: int = 16000,
    duration: float = 3.0,
    n_mfcc: int = 40,
) -> np.ndarray:
    target_length = int(sample_rate * duration)

    audio, _ = librosa.load(file_path, sr=sample_rate)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak

    audio, _ = librosa.effects.trim(audio, top_db=25)

    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    else:
        audio = audio[:target_length]

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feature = np.concatenate([
        np.mean(mfcc.T, axis=0),
        np.mean(delta.T, axis=0),
        np.mean(delta2.T, axis=0),
    ])
    return feature.astype(np.float32)
