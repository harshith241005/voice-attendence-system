"""CLI: train the voice attendance model."""

from __future__ import annotations

from .model import train_model


def main() -> None:
    metrics = train_model()
    print(metrics)


if __name__ == "__main__":
    main()
