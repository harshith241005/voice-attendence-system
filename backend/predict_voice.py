"""CLI: predict live voice and optionally mark attendance."""

from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DURATION, DEFAULT_SAMPLE_RATE
from .service import predict_and_optionally_mark


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict student voice from microphone")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    result = predict_and_optionally_mark(
        duration=args.duration,
        sample_rate=args.sample_rate,
        threshold=args.threshold,
    )
    print(result)


if __name__ == "__main__":
    main()
