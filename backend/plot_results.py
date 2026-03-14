"""CLI: generate training curves from saved history."""

from __future__ import annotations

from .evaluate import plot_training_curves


def main() -> None:
    paths = plot_training_curves()
    print(paths)


if __name__ == "__main__":
    main()
