"""CLI: download speaker dataset into dataset/train and dataset/test."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DATASET_DIR
from .dataset import download_demo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download split speaker voice dataset")
    parser.add_argument("--samples-per-student", type=int, default=40)
    args = parser.parse_args()

    summary = download_demo_dataset(Path(DATASET_DIR), args.samples_per_student)
    print(summary)


if __name__ == "__main__":
    main()
