"""Flask backend API for voice attendance system."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from .config import TEST_DATA_DIR
from .dataset import download_demo_dataset
from .db import init_db, list_attendance, list_students
from .service import predict_file_and_optionally_mark


app = Flask(__name__)
CORS(app)


@app.get("/")
def index() -> tuple:
    return (
        jsonify(
            {
                "service": "voice-attendance-backend",
                "status": "ok",
                "health": "/health",
            }
        ),
        200,
    )


@app.get("/health")
def health() -> tuple:
    try:
        init_db()
        return jsonify({"status": "ok"}), 200
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.get("/students")
def get_students() -> tuple:
    try:
        return jsonify({"students": list_students()}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/attendance")
def get_attendance() -> tuple:
    try:
        limit = int(request.args.get("limit", 500))
        rows = list_attendance(limit=limit)
        payload = [
            {
                "id": row[0],
                "name": row[1],
                "date": row[2],
                "time": row[3],
                "confidence": row[4],
                "source": row[5],
            }
            for row in rows
        ]
        return jsonify({"rows": payload}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/dataset/download")
def post_download_dataset() -> tuple:
    try:
        data = request.get_json(silent=True) or {}
        samples_per_student = int(data.get("samples_per_student", 120))
        summary = download_demo_dataset(Path(Path(TEST_DATA_DIR).parent), samples_per_student)
        return jsonify({"summary": summary}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/train")
def post_train() -> tuple:
    try:
        from .model import train_model

        metrics = train_model()
        return jsonify({"metrics": metrics}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/evaluate")
def post_evaluate() -> tuple:
    try:
        from .evaluate import evaluate_model, plot_training_curves

        metrics = evaluate_model()
        plots = plot_training_curves()
        return jsonify({"metrics": metrics, "plots": plots}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/predict/test-file")
def post_predict_test_file() -> tuple:
    try:
        data = request.get_json(silent=True) or {}
        student = str(data.get("student", "")).strip()
        file_name = str(data.get("file_name", "")).strip()
        threshold = float(data.get("threshold", 0.55))
        expected_name = data.get("expected_name")
        if expected_name == "":
            expected_name = None

        if not student or not file_name:
            return jsonify({"error": "student and file_name are required"}), 400

        base = Path(TEST_DATA_DIR).resolve()
        target = (base / student / file_name).resolve()

        if base not in target.parents:
            return jsonify({"error": "invalid file path"}), 400
        if not target.exists() or not target.is_file():
            return jsonify({"error": "test file not found"}), 404

        result = predict_file_and_optionally_mark(
            file_path=str(target),
            threshold=threshold,
            expected_name=expected_name,
            source="dataset_test",
        )
        return jsonify({"result": result}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/predict/upload")
def post_predict_upload() -> tuple:
    temp_path = ""
    try:
        if "file" not in request.files:
            return jsonify({"error": "file is required"}), 400

        wav_file = request.files["file"]
        if not wav_file.filename:
            return jsonify({"error": "empty filename"}), 400

        threshold = float(request.form.get("threshold", 0.55))
        expected_name = request.form.get("expected_name")
        if expected_name == "":
            expected_name = None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        temp_path = tmp.name
        wav_file.save(temp_path)

        result = predict_file_and_optionally_mark(
            file_path=temp_path,
            threshold=threshold,
            expected_name=expected_name,
            source="uploaded_test_clip",
        )
        return jsonify({"result": result}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
