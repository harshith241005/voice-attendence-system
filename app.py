"""CLI app for voice attendance prediction and marking."""

from __future__ import annotations

from backend.config import DEFAULT_CONFIDENCE_THRESHOLD
from backend.db import init_db
from backend.service import predict_and_optionally_mark


def main() -> None:
    init_db()
    result = predict_and_optionally_mark(threshold=DEFAULT_CONFIDENCE_THRESHOLD)

    print(f"\nRecognized student: {result['name']}")
    print(f"Confidence: {result['confidence']:.2%}")

    if result["attendance_marked"]:
        print(f"Attendance marked for {result['name']}")
    else:
        print(
            "Confidence too low. "
            f"Threshold is {result['threshold']:.2f}, attendance not marked."
        )


if __name__ == "__main__":
    main()
