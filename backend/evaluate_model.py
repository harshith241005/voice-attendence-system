"""CLI: run model evaluation."""

from __future__ import annotations

from .evaluate import evaluate_model


def main() -> None:
    metrics = evaluate_model()
    print({"accuracy": metrics["accuracy"]})


if __name__ == "__main__":
    main()
