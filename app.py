from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

if __name__ == "__main__" and not st.runtime.exists():
    raise SystemExit(subprocess.call([
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())
    ]))

import re
import tempfile

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from input_adapter import adapt_model_input
from pose_pattern_utils import EXERCISE_MAP
from rf_calibration import calibrated_probability
from video_extractor import video_to_visibility_aware_pattern


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "artifacts_uiprmd_fk" / "models"
MODEL_INDEX_PATH = MODEL_DIR / "model_index.json"
MAX_UPLOAD_BYTES = 150 * 1024 * 1024
NEAR_THRESHOLD_MARGIN = 0.08

LOWER_BODY_CONNECTIONS = (
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"), ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"), ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ("LEFT_HIP", "RIGHT_HIP"),
)
UPPER_BODY_CONNECTIONS = (
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
)
TORSO_CONNECTIONS = (
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
)
FULL_BODY_CONNECTIONS = UPPER_BODY_CONNECTIONS + TORSO_CONNECTIONS + LOWER_BODY_CONNECTIONS


@st.cache_data(show_spinner=False)
def load_index():
    if not MODEL_INDEX_PATH.exists():
        raise FileNotFoundError(f"Indeks model tidak ditemukan: {MODEL_INDEX_PATH}")
    return json.loads(MODEL_INDEX_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_artifact(filename):
    path = MODEL_DIR / Path(filename).name
    if not path.exists():
        raise FileNotFoundError(f"Artefak model tidak ditemukan: {path.name}")
    return joblib.load(path)


def prediction_label(probability, threshold):
    return "Correct" if probability >= threshold else "Incorrect"


def draw_exercise_pose(frame, result, exercise_code):
    if not result.pose_landmarks:
        return frame
    landmarks = result.pose_landmarks.landmark
    height, width = frame.shape[:2]
    pose_landmark = mp.solutions.pose.PoseLandmark

    def position(name):
        landmark = landmarks[getattr(pose_landmark, name)]
        return (int(landmark.x * width), int(landmark.y * height)), float(landmark.visibility)

    overlay = frame.copy()
    show_full_body = exercise_code == "m07"
    displayed = FULL_BODY_CONNECTIONS if show_full_body else LOWER_BODY_CONNECTIONS
    primary = UPPER_BODY_CONNECTIONS if exercise_code == "m07" else LOWER_BODY_CONNECTIONS
    primary_set = set(primary)
    primary_nodes = {name for pair in primary for name in pair}

    for first, second in displayed:
        point_a, visibility_a = position(first)
        point_b, visibility_b = position(second)
        if min(visibility_a, visibility_b) >= 0.35:
            is_primary = (first, second) in primary_set
            color = (71, 208, 145) if is_primary else (175, 181, 178)
            cv2.line(overlay, point_a, point_b, color, 5 if is_primary else 3, cv2.LINE_AA)
    for name in sorted({name for pair in displayed for name in pair}):
        point, visibility = position(name)
        if visibility >= 0.35:
            cv2.circle(overlay, point, 7, (255, 255, 255), -1, cv2.LINE_AA)
            color = (35, 122, 87) if name in primary_nodes else (115, 123, 119)
            cv2.circle(overlay, point, 7, color, 2, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)


def attach_evidence(video_path, candidates, repetitions, exercise_code):
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    pose_module = mp.solutions.pose
    try:
        with pose_module.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            for repetition, candidate in zip(repetitions, candidates):
                evidence = []
                for item in candidate.get("keyframes", []):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(item["source_frame"]))
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    pose_result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame = draw_exercise_pose(frame, pose_result, exercise_code)
                    evidence_item = dict(item)
                    evidence_item["image"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    evidence.append(evidence_item)
                repetition["evidence"] = evidence
    finally:
        capture.release()


def analyze_video(video_path, original_name, exercise):
    index = load_index()
    if exercise not in index["exercises"]:
        raise ValueError(f"Model gerakan {exercise} tidak tersedia.")

    _, _, _, _, _, diagnostics, candidates = video_to_visibility_aware_pattern(
        video_path, exercise, target_steps=80
    )
    mode = "one_side" if diagnostics["view_mode"] == "one_side" else "standard"
    info = index["exercises"][exercise]["modes"][mode]
    model = load_artifact(info["model_path"])
    calibrator = load_artifact(info["calibrator_path"])
    threshold = float(info["threshold"])
    repetitions = []

    for number, candidate in enumerate(candidates, start=1):
        adapted_row, adapter_details = adapt_model_input(
            candidate["row"], exercise, mode, info["feature_columns"]
        )
        frame = pd.DataFrame(
            [adapted_row], columns=info["feature_columns"]
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_probability = float(model.predict_proba(frame)[0, 1])
        probability = float(calibrated_probability(calibrator, [raw_probability])[0])
        repetitions.append({
            "number": number,
            "status": prediction_label(probability, threshold),
            "raw_probability": raw_probability,
            "probability": probability,
            "threshold": threshold,
            "margin": probability - threshold,
            "near_threshold": abs(probability - threshold) <= NEAR_THRESHOLD_MARGIN,
            "start": candidate.get("source_start", candidate["start"]),
            "end": candidate.get("source_end", candidate["end"]),
            "frames": candidate["used_frames"],
            "model_input_adapter": adapter_details,
        })

    if not repetitions:
        raise RuntimeError("Tidak ada repetisi yang dapat dinilai.")
    attach_evidence(video_path, candidates, repetitions, exercise)

    median_probability = float(np.median([item["probability"] for item in repetitions]))
    near_threshold = abs(median_probability - threshold) <= NEAR_THRESHOLD_MARGIN
    dominant_side = (
        diagnostics.get("dominant_arm") if exercise == "m07"
        else diagnostics.get("dominant_leg")
    ) or "kedua sisi"
    feature_bases = sorted({
        re.split(r"_t\d{3}|__", column)[0] for column in info["feature_columns"]
    })
    return {
        "video_name": original_name,
        "exercise_code": exercise,
        "exercise_name": EXERCISE_MAP[exercise],
        "status": prediction_label(median_probability, threshold),
        "near_threshold": near_threshold,
        "median_probability": median_probability,
        "threshold": threshold,
        "margin": median_probability - threshold,
        "mode": mode,
        "mode_label": "Tampak satu sisi" if mode == "one_side" else "Tampak depan/serong",
        "dominant_side": dominant_side,
        "pose_detection_ratio": diagnostics["pose_detection_ratio"],
        "one_side_ratio": diagnostics["one_side_frame_ratio"],
        "detected_frames": diagnostics["detected_frames"],
        "total_frames": diagnostics["total_frames"],
        "feature_profile": info.get("feature_profile", "temporal_summary"),
        "feature_bases": feature_bases,
        "masked_features": diagnostics.get("masked_features", []),
        "input_adapter_version": diagnostics.get("input_adapter_version"),
        "input_adapter_features": diagnostics.get("input_adapter_features", []),
        "repetitions": repetitions,
    }


def render_result(result):
    status_class = "correct" if result["status"] == "Correct" else "incorrect"
    st.markdown(
        f"""
        <section class="result-title">
          <div><small>HASIL ANALISIS · {result['exercise_code'].upper()}</small>
          <h2>{result['exercise_name']}</h2></div>
          <div class="status-pill {status_class}"><span>Keputusan RF</span><b>{result['status']}</b></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(result["video_name"])
    if result["near_threshold"]:
        st.warning(
            "Hasil berada dekat threshold Kinect-only. Pastikan seluruh tubuh dan satu repetisi "
            "terlihat lengkap."
        )

    first, second, third, fourth = st.columns(4)
    first.metric("Probabilitas Correct", f"{result['median_probability'] * 100:.1f}%")
    second.metric("Threshold Kinect", f"{result['threshold']:.3f}", f"{result['margin']:+.3f}")
    third.metric("Mode kamera", result["mode_label"], f"Sisi: {result['dominant_side']}")
    fourth.metric(
        "Deteksi pose",
        f"{result['pose_detection_ratio'] * 100:.1f}%",
        f"{result['detected_frames']} / {result['total_frames']} frame",
    )
    st.progress(float(np.clip(result["median_probability"], 0.0, 1.0)))

    st.subheader("Hasil per repetisi")
    table = pd.DataFrame([{
        "Repetisi": f"#{item['number']}",
        "Frame": f"{item['start']}–{item['end']}",
        "P(Correct)": f"{item['probability'] * 100:.1f}%",
        "Threshold": f"{item['threshold']:.3f}",
        "Hasil": item["status"],
    } for item in result["repetitions"]])
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Potongan gerakan krusial")
    if result["exercise_code"] == "m07":
        st.caption("Seluruh tubuh ditampilkan dengan fokus pada bahu, siku, dan tangan.")
    else:
        st.caption("Garis hijau menekankan pinggul, lutut, dan kaki.")
    st.caption("Badge keyframe adalah pemeriksaan geometri, bukan prediksi RF per frame.")

    for repetition in result["repetitions"]:
        st.markdown(
            f"**Repetisi #{repetition['number']}** — RF sekuens: "
            f"{repetition['status']} · {repetition['probability'] * 100:.1f}%"
        )
        evidence = repetition.get("evidence", [])
        columns = st.columns(max(1, len(evidence)))
        for column, item in zip(columns, evidence):
            with column:
                st.image(item["image"], use_container_width=True)
                st.markdown(f"**{item['phase']}** · frame {item['source_frame']}")
                tone = item.get("tone", "neutral")
                message = f"{item.get('assessment', '')}\n\n{item.get('detail', '')}"
                if tone == "good":
                    st.success(message)
                elif tone == "review":
                    st.warning(message)
                else:
                    st.info(message)
                metrics = item.get("metrics", {})
                if metrics:
                    st.caption(" · ".join(f"{name}: {value}" for name, value in metrics.items()))

    with st.expander("Detail teknis"):
        st.write({
            "profil_fitur": result["feature_profile"],
            "mode_model": result["mode"],
            "rasio_satu_sisi": result["one_side_ratio"],
            "fitur_dimask": result["masked_features"],
            "cakupan_fitur": result["feature_bases"],
            "input_adapter": result["input_adapter_version"],
            "fitur_adapter": result["input_adapter_features"],
            "adapter_per_repetisi": [
                item["model_input_adapter"] for item in result["repetitions"]
            ],
        })


def main():
    st.set_page_config(
        page_title="MotionLens RF",
        page_icon="🧍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #f5f2e9; color: #17241e; }
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
        .hero { padding: 1.5rem 1.7rem; border: 1px solid #d9dfd9; border-radius: 22px;
                background: #fffdf8; margin-bottom: 1rem; }
        .hero small, .result-title small { color: #087b59; font-weight: 800; letter-spacing: .12em; }
        .hero h1 { margin: .2rem 0 .4rem; font-family: Georgia, serif; }
        .hero p { margin: 0; color: #53645c; }
        .result-title { display:flex; justify-content:space-between; align-items:center;
                        margin-top:2rem; padding-top:1rem; border-top:1px solid #d8ded9; }
        .result-title h2 { margin:.15rem 0 0; font-family: Georgia, serif; }
        .status-pill { padding:.7rem 1.2rem; border-radius:16px; text-align:right; }
        .status-pill span { display:block; font-size:.78rem; }
        .status-pill b { font-size:1.35rem; }
        .status-pill.correct { background:#dcefe6; color:#075f47; }
        .status-pill.incorrect { background:#f6deda; color:#8e2f23; }
        div[data-testid="stMetric"] { background:#fffdf8; border:1px solid #d8ded9;
                                      padding:1rem; border-radius:16px; min-height:145px; }
        div[data-testid="stFileUploader"] { background:#fffdf8; border-radius:16px; padding:.7rem; }
        .stButton > button { width:100%; border-radius:12px; background:#087b59; color:white;
                             border:0; font-weight:700; min-height:3rem; }
        .stButton > button:hover { background:#056447; color:white; border:0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="hero">
          <small>MOTIONLENS · RANDOM FOREST KINECT-ONLY</small>
          <h1>Analisis kualitas gerakan dari video</h1>
          <p>MediaPipe mengekstrak pola tubuh, kemudian Random Forest menilai satu repetisi penuh.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    input_column, preview_column = st.columns([1, 1.35], gap="large")
    with input_column:
        exercise = st.selectbox(
            "Jenis gerakan",
            options=list(EXERCISE_MAP),
            format_func=lambda code: f"{code.upper()} — {EXERCISE_MAP[code]}",
        )
        uploaded = st.file_uploader(
            "Unggah video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Maksimal 150 MB. Pastikan seluruh tubuh dan satu repetisi terlihat.",
        )
        analyze = st.button("Analisis video", type="primary", disabled=uploaded is None)
        st.caption("Model: Random Forest · Training dan threshold: Kinect UI-PRMD")
    with preview_column:
        if uploaded is not None:
            st.video(uploaded.getvalue())
        else:
            st.info("Preview video akan tampil setelah file dipilih.")

    if analyze and uploaded is not None:
        video_bytes = uploaded.getvalue()
        if len(video_bytes) > MAX_UPLOAD_BYTES:
            st.error("Ukuran video melebihi batas 150 MB.")
            return
        suffix = Path(uploaded.name).suffix.lower() or ".mp4"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary.write(video_bytes)
                temporary_path = Path(temporary.name)
            with st.spinner("Mendeteksi pose, memisahkan repetisi, dan menjalankan Random Forest..."):
                result = analyze_video(temporary_path, uploaded.name, exercise)
            st.session_state["analysis_result"] = result
        except Exception as error:
            st.session_state.pop("analysis_result", None)
            st.error(f"Analisis gagal: {error}")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    result = st.session_state.get("analysis_result")
    if result:
        render_result(result)


if __name__ == "__main__":
    main()
