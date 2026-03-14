"""CLI: record student dataset samples."""

from __future__ import annotations

import argparse

from .audio import record_dataset_samples
from .config import DEFAULT_DURATION, DEFAULT_SAMPLE_RATE, TRAIN_DATA_DIR
from .db import seed_students


def main() -> None:
    parser = argparse.ArgumentParser(description="Record student voice samples")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    args = parser.parse_args()

    saved = record_dataset_samples(
        name=args.name,
        count=args.count,
        dataset_dir=TRAIN_DATA_DIR,
        duration=args.duration,
        sample_rate=args.sample_rate,
    )
    seed_students([args.name])
    print(f"Saved {len(saved)} files for {args.name}")


if __name__ == "__main__":
    main()
