"""Streamlit frontend for voice attendance system (Flask API client)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DATA_DIR = DATASET_DIR / "train"
TEST_DATA_DIR = DATASET_DIR / "test"
RESULTS_DIR = BASE_DIR / "results"
DB_PATH = BASE_DIR / "database" / "attendance.db"
MODEL_PATH = BASE_DIR / "models" / "voice_model.h5"
EVAL_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.json"
TARGET_STUDENTS = ["Likith", "Sateesh", "Raghu", "Harshith"]

API_BASE_URL = os.getenv("VOICE_API_URL", "http://127.0.0.1:5000").rstrip("/")
API_TIMEOUT_SHORT = 120
API_TIMEOUT_LONG = 1800


def _api_get(path: str, params: dict | None = None) -> dict:
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.get(url, params=params, timeout=300)
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            return {"ok": False, "error": data.get("error", f"HTTP {resp.status_code}")}
        return {"ok": True, "data": data}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def _api_post_json(path: str, payload: dict) -> dict:
    url = f"{API_BASE_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=API_TIMEOUT_LONG)
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            return {"ok": False, "error": data.get("error", f"HTTP {resp.status_code}")}
        return {"ok": True, "data": data}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def _api_post_file(path: str, file_name: str, file_bytes: bytes, form_data: dict[str, str]) -> dict:
    url = f"{API_BASE_URL}{path}"
    files = {"file": (file_name, file_bytes, "audio/wav")}
    try:
        resp = requests.post(url, files=files, data=form_data, timeout=API_TIMEOUT_SHORT)
        data = resp.json() if resp.content else {}
        if resp.status_code >= 400:
            return {"ok": False, "error": data.get("error", f"HTTP {resp.status_code}")}
        return {"ok": True, "data": data}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


@st.cache_data(ttl=120, show_spinner=False)
def _cached_health() -> dict:
    return _api_get("/health")


@st.cache_data(ttl=120, show_spinner=False)
def _cached_students() -> dict:
    return _api_get("/students")


def _student_set() -> list[str]:
    students_resp = _cached_students()
    api_students = students_resp["data"].get("students", []) if students_resp["ok"] else []
    if api_students:
        return api_students

    local_students = sorted([p.name for p in TRAIN_DATA_DIR.glob("*") if p.is_dir()])
    return local_students if local_students else TARGET_STUDENTS


def _render_prediction_result(result: dict, actual_label: str | None = None) -> None:
    st.success(f"Recognized: {result['name']} ({result['confidence']:.2%})")

    if actual_label and actual_label != "Unknown":
        if result["name"] == actual_label:
            st.info(f"Prediction matches provided actual label: {actual_label}")
        else:
            st.warning(
                f"Prediction does not match provided actual label. "
                f"Actual: {actual_label}, Predicted: {result['name']}"
            )

    if result["attendance_marked"]:
        st.info("Attendance marked in SQLite database.")
    else:
        if result.get("block_reason") == "low_confidence":
            st.warning("Confidence below threshold; attendance not marked.")
        elif result.get("block_reason") == "expected_name_mismatch":
            st.warning(
                f"Predicted '{result['name']}' but expected '{result.get('expected_name')}'. "
                "Attendance not marked."
            )
        else:
            st.warning("Attendance not marked.")

    prob_df = pd.DataFrame(
        [{"Student": k, "Probability": v} for k, v in result["all_probs"].items()]
    ).sort_values("Probability", ascending=False)
    st.bar_chart(prob_df.set_index("Student"))


st.set_page_config(page_title="Voice Attendance", page_icon="🎤", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --accent: #0f766e;
        --accent-soft: #ccfbf1;
        --surface: #f8fafc;
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 80% -10%, #d1fae5 0%, rgba(209,250,229,0) 60%),
            radial-gradient(900px 500px at -10% 20%, #e0f2fe 0%, rgba(224,242,254,0) 55%),
            var(--surface);
    }
    .hero {
        border: 1px solid #cbd5e1;
        background: linear-gradient(135deg, #ecfeff 0%, #f0fdfa 45%, #f8fafc 100%);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .chip {
        display: inline-block;
        border: 1px solid #99f6e4;
        background: #f0fdfa;
        color: #115e59;
        border-radius: 999px;
        padding: 0.1rem 0.6rem;
        margin-right: 0.4rem;
        font-size: 0.82rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0 0 0.35rem 0; color:#134e4a;">Voice Attendance Dashboard</h2>
      <div style="margin-bottom:0.55rem; color:#334155;">Flask API backend + NNDL speaker classifier with split train/test workflow.</div>
      <span class="chip">Likith</span>
      <span class="chip">Sateesh</span>
      <span class="chip">Raghu</span>
      <span class="chip">Harshith</span>
    </div>
    """,
    unsafe_allow_html=True,
)

health = _cached_health()
if not health["ok"]:
    st.error(f"Flask backend unavailable at {API_BASE_URL}: {health['error']}")
    st.stop()

students = _student_set()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Backend", "Connected")
with col2:
    st.metric("Database", "Ready" if DB_PATH.exists() else "Missing")
with col3:
    st.metric("Model", "Ready" if MODEL_PATH.exists() else "Not trained")
with col4:
    student_count = len([p for p in TRAIN_DATA_DIR.glob("*") if p.is_dir()])
    st.metric("Dataset Students", student_count)

tab1, tab2, tab3, tab4 = st.tabs(["Attendance", "Dataset and Training", "Evaluation", "Attendance Logs"])

with tab1:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Predict and Mark Attendance")
        expected_options = ["Any student"] + students if students else ["Any student"]
        selected_expected = st.selectbox(
            "Expected student (optional)",
            options=expected_options,
            index=0,
            help="Marks attendance only when prediction matches this student.",
        )
        threshold = st.slider("Confidence threshold", min_value=0.3, max_value=0.95, value=0.55, step=0.01)
        source_mode = st.radio(
            "Clip source",
            options=["Held-out test dataset", "Upload test clip"],
            horizontal=True,
        )

    with right:
        st.subheader("Quick Notes")
        st.markdown("- Use train/test split only")
        st.markdown("- No microphone mode")
        st.markdown("- Supports drag-and-drop WAV")
        st.markdown("- Attendance saved in SQLite")

    expected_name = None if selected_expected == "Any student" else selected_expected

    if source_mode == "Held-out test dataset":
        st.markdown("### Select Existing Test Clip")
        available_students = sorted([p.name for p in TEST_DATA_DIR.glob("*") if p.is_dir()])

        if not available_students:
            st.warning("No test dataset found. Download dataset first in Dataset and Training tab.")
        else:
            a, b, c = st.columns([1, 1, 1.2])
            with a:
                chosen_student = st.selectbox("Student", options=available_students)
            student_files = sorted((TEST_DATA_DIR / chosen_student).glob("*.wav"))
            file_names = [p.name for p in student_files]
            with b:
                selected_file = st.selectbox("Test clip", options=file_names if file_names else ["No files"])
            with c:
                trigger = st.button("Predict and Mark from Test Data", use_container_width=True)

            if file_names and trigger:
                with st.spinner("Predicting from held-out test file..."):
                    resp = _api_post_json(
                        "/predict/test-file",
                        {
                            "student": chosen_student,
                            "file_name": selected_file,
                            "threshold": threshold,
                            "expected_name": expected_name,
                        },
                    )

                if not resp["ok"]:
                    st.error(resp["error"])
                else:
                    st.caption(f"Ground truth folder: {chosen_student} | Test file: {selected_file}")
                    _render_prediction_result(resp["data"]["result"])

    else:
        st.markdown("### Drag and Drop Test Clip")
        up1, up2 = st.columns([1.2, 1])
        with up1:
            uploaded_clip = st.file_uploader(
                "Upload one WAV test clip",
                type=["wav"],
                key="test_clip_uploader",
                help="Drag and drop WAV clip here.",
            )
        with up2:
            actual_label_options = ["Unknown"] + students if students else ["Unknown"]
            actual_label = st.selectbox(
                "Actual person (optional)",
                options=actual_label_options,
                index=0,
                help="For demo verification only.",
            )

        if uploaded_clip is not None and st.button("Predict and Mark from Uploaded Clip", use_container_width=True):
            with st.spinner("Predicting from uploaded clip..."):
                resp = _api_post_file(
                    "/predict/upload",
                    uploaded_clip.name,
                    uploaded_clip.getvalue(),
                    {
                        "threshold": str(threshold),
                        "expected_name": expected_name or "",
                    },
                )

            if not resp["ok"]:
                st.error(resp["error"])
            else:
                _render_prediction_result(resp["data"]["result"], actual_label=actual_label)

with tab2:
    st.subheader("Dataset and Training")
    st.caption("Configured students: Likith, Sateesh, Raghu, Harshith")

    c1, c2 = st.columns([1, 1])
    with c1:
        samples = st.number_input("Internet samples per student", min_value=60, max_value=400, value=120, step=20)
    with c2:
        st.caption("Dataset layout")
        st.code("dataset/train/<student>\ndataset/test/<student>")

    if st.button("Download and Prepare Dataset", use_container_width=True):
        with st.spinner("Downloading and splitting dataset..."):
            resp = _api_post_json("/dataset/download", {"samples_per_student": int(samples)})

        if not resp["ok"]:
            st.error(resp["error"])
        else:
            st.success("Dataset prepared successfully with train/test split.")
            st.write(resp["data"]["summary"])

    if st.button("Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            resp = _api_post_json("/train", {})

        if not resp["ok"]:
            st.error(resp["error"])
        else:
            metrics = resp["data"]["metrics"]
            st.success(f"Training complete. Test accuracy: {metrics['test_accuracy']:.2%}")

with tab3:
    st.subheader("Evaluation")

    if st.button("Run Full Evaluation", use_container_width=True):
        with st.spinner("Running evaluation and plots..."):
            resp = _api_post_json("/evaluate", {})

        if not resp["ok"]:
            st.error(resp["error"])
        else:
            metrics = resp["data"]["metrics"]
            plots = resp["data"]["plots"]
            st.success(f"Evaluation complete. Accuracy: {metrics['accuracy']:.2%}")
            st.write("Saved plots:", plots)

    if EVAL_METRICS_PATH.exists():
        st.caption("Latest metrics: results/evaluation_metrics.json")

    img_cols = st.columns(3)
    images = ["confusion_matrix.png", "accuracy_curve.png", "loss_curve.png"]
    for idx, img in enumerate(images):
        img_path = RESULTS_DIR / img
        if img_path.exists():
            with img_cols[idx]:
                st.image(str(img_path), caption=img, width="stretch")

with tab4:
    st.subheader("Attendance Database")
    students_label = ", ".join(students) if students else "None"
    st.write(f"Registered students: {students_label}")

    attendance_resp = _api_get("/attendance", params={"limit": 500})
    if not attendance_resp["ok"]:
        st.error(attendance_resp["error"])
    else:
        rows = attendance_resp["data"].get("rows", [])
        if rows:
            df = pd.DataFrame(rows, columns=["id", "name", "date", "time", "confidence", "source"])
            st.dataframe(df, width="stretch")
            st.download_button(
                "Download Attendance CSV",
                data=df.to_csv(index=False),
                file_name="attendance_export.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No attendance records yet.")
