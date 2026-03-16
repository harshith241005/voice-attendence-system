# Deep Learning Voice Attendance System (Complete Medium-Level MVP)

This project uses a Flask backend API, a Streamlit frontend client, and SQLite for attendance storage.

## Final Architecture

Downloaded Speaker Dataset -> Train/Test Split -> MFCC + Delta Features -> Deep Learning Model -> Test-Clip Prediction -> SQLite Attendance DB -> Evaluation/Graphs

## Technologies

- Python
- TensorFlow / Keras
- librosa
- sounddevice
- SQLite
- numpy, pandas
- scikit-learn
- matplotlib
- streamlit
- flask
- flask-cors
- requests

## Project Structure

```text
voice-attendance-system/
├── app.py
├── requirements.txt
├── README.md
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── audio.py
│   ├── features.py
│   ├── dataset.py
│   ├── model.py
│   ├── service.py
│   ├── db.py
│   ├── evaluate.py
│   └── flask_app.py
├── frontend/
│   └── streamlit_app.py
├── dataset/
│   ├── train/
│   │   ├── likith/
│   │   ├── rahul/
│   │   └── sneha/
│   └── test/
│       ├── likith/
│       ├── rahul/
│       └── sneha/
├── models/
│   ├── voice_model.h5
│   └── labels.json
├── database/
│   └── attendance.db
└── results/
    ├── training_metadata.json
    ├── evaluation_metrics.json
    ├── accuracy_curve.png
    ├── loss_curve.png
    └── confusion_matrix.png
```

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

### 1) Download Internet Dataset with Separate Train/Test

```bash
python -m backend.download_dataset --samples-per-student 120
```

This creates split data for:
- dataset/train/Likith, dataset/train/Sateesh, dataset/train/Raghu, dataset/train/Harshith
- dataset/test/Likith, dataset/test/Sateesh, dataset/test/Raghu, dataset/test/Harshith

Training uses only dataset/train, and evaluation uses dataset/test.

### 2) Train Model

```bash
python -m backend.train_model
```

### 3) Run Evaluation and Plots

```bash
python -m backend.evaluate_model
python -m backend.plot_results
```

### 4) Start Flask Backend API

```bash
python -m backend.flask_app
```

### 5) Run Frontend Dashboard

```bash
set VOICE_API_URL=http://127.0.0.1:5000
python -m streamlit run frontend/streamlit_app.py
```

In the app, use Held-out test dataset mode under Attendance Prediction to run demo predictions and attendance marking from test samples.

## Deploy Frontend and Backend on One Provider (Render)

This repository includes a ready blueprint file at `render.yaml` that creates:
- `voice-attendance-backend` (Flask API)
- `voice-attendance-frontend` (Streamlit UI)

### A) Push code to GitHub

1. Create a GitHub repository and push this project.
2. Keep `render.yaml` in the project root.

### B) Create both services in one Render project

1. Sign in to Render.
2. Click **New** -> **Blueprint**.
3. Select your GitHub repository.
4. Render reads `render.yaml` and creates both services automatically.
5. Deploy.

### C) Connect frontend to backend URL

After backend deploy completes, copy backend URL:
- Example: `https://voice-attendance-backend.onrender.com`

Set frontend env var:
- Key: `VOICE_API_URL`
- Value: backend URL

Then redeploy the frontend service.

### D) Verify

1. Open backend health endpoint:
    - `https://<your-backend>/health`
    - Should return status ok.
2. Open frontend URL.
3. Frontend should show backend connected.

### Notes

- Render free plan sleeps on inactivity, so first request can be slow.
- Attendance DB on free web service is ephemeral. If persistence is required, use Render persistent disk or managed database.
- If model files are large, first deploy/build can take longer.

## SQLite Database Details

Database path:
- `database/attendance.db`

Tables:
- `students(id, name)`
- `attendance(id, student_id, name, date, time, confidence, source)`

Attendance is marked when prediction confidence >= 0.55 by default.

## Notes

- Database file is created automatically by backend initialization.
- Project supports full downloaded dataset workflow with separate train/test usage.
- On Windows, allow microphone access in Privacy settings if recording fails.
