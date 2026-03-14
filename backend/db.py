"""SQLite operations for students and attendance."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime

from .config import DB_PATH, DATABASE_DIR


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(DATABASE_DIR, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: str = DB_PATH) -> None:
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        conn.commit()


def seed_students(names: list[str], db_path: str = DB_PATH) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.cursor()
        for name in sorted(set(names)):
            cur.execute("INSERT OR IGNORE INTO students (name) VALUES (?)", (name,))
        conn.commit()


def _get_student_id(name: str, db_path: str = DB_PATH) -> int:
    seed_students([name], db_path)
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM students WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Student not found after insert: {name}")
        return int(row[0])


def mark_attendance(
    name: str,
    confidence: float,
    source: str = "microphone",
    db_path: str = DB_PATH,
) -> None:
    init_db(db_path)
    sid = _get_student_id(name, db_path)

    now = datetime.now()
    date_value = now.strftime("%Y-%m-%d")
    time_value = now.strftime("%H:%M:%S")

    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO attendance (student_id, name, date, time, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sid, name, date_value, time_value, confidence, source),
        )
        conn.commit()


def list_attendance(limit: int = 200, db_path: str = DB_PATH) -> list[tuple]:
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, date, time, confidence, source
            FROM attendance
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def list_students(db_path: str = DB_PATH) -> list[str]:
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM students ORDER BY name")
        return [row[0] for row in cur.fetchall()]
