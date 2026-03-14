"""CLI: show students and recent attendance rows."""

from __future__ import annotations

from .db import init_db, list_attendance, list_students


def main() -> None:
    init_db()
    print("Students:", list_students())
    print("Recent attendance rows:")
    for row in list_attendance(limit=20):
        print(row)


if __name__ == "__main__":
    main()
