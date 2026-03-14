"""Microphone recording helpers."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


def record_temp_wav(duration: float, sample_rate: int) -> str:
    """Record one clip and return a temp wav path."""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()

    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    write(tmp.name, sample_rate, audio_int16)
    return tmp.name


def record_dataset_samples(
    name: str,
    count: int,
    dataset_dir: str,
    duration: float,
    sample_rate: int,
) -> list[str]:
    folder = os.path.join(dataset_dir, name)
    os.makedirs(folder, exist_ok=True)

    saved = []
    for i in range(1, count + 1):
        input(f"Press Enter and speak for sample {i}/{count}...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        path = os.path.join(folder, f"{name}{i}.wav")
        write(path, sample_rate, audio_int16)
        saved.append(path)
    return saved
