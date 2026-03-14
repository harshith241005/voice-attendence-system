"""Speaker dataset download utility."""

from __future__ import annotations

import shutil
from pathlib import Path
from collections import Counter

from tensorflow.keras.utils import get_file


FSDD_URL = (
    "https://github.com/Jakobovski/free-spoken-digit-dataset/"
    "archive/refs/heads/master.zip"
)


def _resolve_recordings_root(archive_path: str, dataset_dir: Path) -> Path:
    base = Path(archive_path).with_suffix("")
    candidates = [
        base / "free-spoken-digit-dataset-master" / "recordings",
        dataset_dir.parent / "downloads" / "datasets" / "free-spoken-digit-dataset-master" / "recordings",
        dataset_dir.parent / "downloads" / "free-spoken-digit-dataset-master" / "recordings",
    ]

    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.wav")):
            return candidate

    for candidate in dataset_dir.parent.glob("downloads/**/recordings"):
        if candidate.exists() and any(candidate.glob("*.wav")):
            return candidate

    raise FileNotFoundError("Could not locate FSDD recordings directory.")


def _copy_limited_wavs(source_dir: Path, target_dir: Path, limit: int) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(source_dir.glob("*.wav"))[:limit]
    for index, wav_path in enumerate(wav_files, start=1):
        shutil.copy2(wav_path, target_dir / f"sample_{index:03d}.wav")
    return len(wav_files)


def _speaker_from_filename(wav_name: str) -> str:
    # FSDD pattern: {digitLabel}_{speaker}_{index}.wav
    stem = Path(wav_name).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return "unknown"
    return parts[1]


def _index_from_filename(wav_name: str) -> int:
    stem = Path(wav_name).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return -1
    try:
        return int(parts[2])
    except ValueError:
        return -1


def download_demo_dataset(dataset_dir: Path, samples_per_student: int = 120) -> dict[str, dict[str, int]]:
    archive_path = get_file(
        fname="free_spoken_digit_dataset.zip",
        origin=FSDD_URL,
        extract=True,
        cache_dir=str(dataset_dir.parent),
        cache_subdir="downloads",
    )
    root = _resolve_recordings_root(archive_path, dataset_dir)

    all_wavs = sorted(root.glob("*.wav"))
    if not all_wavs:
        raise ValueError("No WAV files found in downloaded FSDD dataset")

    speaker_counts = Counter(_speaker_from_filename(w.name) for w in all_wavs)
    top_speakers = [spk for spk, _ in speaker_counts.most_common(4)]
    if len(top_speakers) < 4:
        raise ValueError("Need at least 4 speakers from source dataset")

    mapping = {
        "Likith": top_speakers[0],
        "Sateesh": top_speakers[1],
        "Raghu": top_speakers[2],
        "Harshith": top_speakers[3],
    }

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Remove old class folders to keep only current configured students.
    for old_dir in train_dir.iterdir() if train_dir.exists() else []:
        if old_dir.is_dir():
            shutil.rmtree(old_dir)
    for old_dir in test_dir.iterdir() if test_dir.exists() else []:
        if old_dir.is_dir():
            shutil.rmtree(old_dir)

    summary: dict[str, dict[str, int]] = {}
    for student, speaker in mapping.items():
        student_train_dir = train_dir / student
        student_test_dir = test_dir / student

        if student_train_dir.exists():
            shutil.rmtree(student_train_dir)
        if student_test_dir.exists():
            shutil.rmtree(student_test_dir)

        student_train_dir.mkdir(parents=True, exist_ok=True)
        student_test_dir.mkdir(parents=True, exist_ok=True)

        speaker_files = [w for w in all_wavs if _speaker_from_filename(w.name) == speaker]

        selected = speaker_files[:samples_per_student]
        train_count = 0
        test_count = 0
        train_idx = 1
        test_idx = 1

        for wav_path in selected:
            sample_index = _index_from_filename(wav_path.name)
            # Deterministic hold-out split: around 20% to test.
            is_test = sample_index >= 0 and sample_index % 5 == 0

            if is_test:
                shutil.copy2(wav_path, student_test_dir / f"sample_{test_idx:03d}.wav")
                test_idx += 1
                test_count += 1
            else:
                shutil.copy2(wav_path, student_train_dir / f"sample_{train_idx:03d}.wav")
                train_idx += 1
                train_count += 1

        summary[student] = {"train": train_count, "test": test_count}

    return summary
