from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
TEMPLATE_MATCHING_DIR = PROJECT_DIR / "template_matching_multi_angle"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(TEMPLATE_MATCHING_DIR))

from pose_pattern_utils import EXERCISE_MAP  # noqa: E402
from rf_calibration import calibrated_probability  # noqa: E402
from input_adapter import adapt_model_input  # noqa: E402
from video_extractor import video_to_visibility_aware_pattern  # noqa: E402


MODEL_INDEX_PATH = PROJECT_DIR / "artifacts_uiprmd_fk/models/model_index.json"
MODEL_DIR = MODEL_INDEX_PATH.parent
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
NEAR_THRESHOLD_MARGIN = 0.08
GENERATED_DIR = MODULE_DIR / "static/generated"
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
VISUAL_FOCUS = {
    "m07": "Seluruh tubuh ditampilkan; garis hijau menekankan bahu, siku, dan tangan untuk Shoulder Abduction.",
}

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    TEMPLATES_AUTO_RELOAD=True,
)


@lru_cache(maxsize=1)
def load_index():
    if not MODEL_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Model belum tersedia. Jalankan 01_build_randomized_features.py dan "
            "02_train_rf_kinect_only.py terlebih dahulu."
        )
    return json.loads(MODEL_INDEX_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=20)
def load_artifact(path):
    configured_path = Path(path)
    candidates = (configured_path, MODEL_DIR / configured_path.name)
    for candidate in candidates:
        if candidate.exists():
            return joblib.load(candidate)
    raise FileNotFoundError(
        f"Artefak model tidak ditemukan: {configured_path.name}. "
        f"Folder yang diperiksa: {MODEL_DIR}"
    )


def validate_upload(video):
    if video is None or not video.filename:
        raise ValueError("Pilih video yang akan dianalisis.")
    suffix = Path(secure_filename(video.filename)).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Format video harus MP4, AVI, MOV, MKV, atau WEBM.")
    return suffix


def prediction_label(probability, threshold):
    return "Correct" if probability >= threshold else "Incorrect"


def _draw_exercise_pose(frame, result, exercise_code):
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
    displayed_connections = FULL_BODY_CONNECTIONS if show_full_body else LOWER_BODY_CONNECTIONS
    primary_connections = UPPER_BODY_CONNECTIONS if exercise_code == "m07" else LOWER_BODY_CONNECTIONS
    primary_set = set(primary_connections)
    primary_nodes = {item for pair in primary_connections for item in pair}

    # Gambar bagian pendukung lebih tipis agar seluruh postur tetap terlihat.
    for first, second in displayed_connections:
        point_a, visibility_a = position(first)
        point_b, visibility_b = position(second)
        if min(visibility_a, visibility_b) >= 0.35:
            primary = (first, second) in primary_set
            color = (71, 208, 145) if primary else (175, 181, 178)
            width_px = 5 if primary else 3
            cv2.line(overlay, point_a, point_b, color, width_px, cv2.LINE_AA)
    for name in sorted({item for pair in displayed_connections for item in pair}):
        point, visibility = position(name)
        if visibility >= 0.35:
            cv2.circle(overlay, point, 7, (255, 255, 255), -1, cv2.LINE_AA)
            color = (35, 122, 87) if name in primary_nodes else (115, 123, 119)
            cv2.circle(overlay, point, 7, color, 2, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)


def generate_evidence_images(video_path, candidates, repetitions, exercise_code):
    """Create visual keyframes. Their badges are transparent geometry checks, not RF outputs."""
    token = uuid.uuid4().hex
    output_dir = GENERATED_DIR / token
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    pose_module = mp.solutions.pose

    with pose_module.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        for repetition, candidate in zip(repetitions, candidates):
            evidence = []
            for key_number, item in enumerate(candidate.get("keyframes", []), start=1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(item["source_frame"]))
                ok, frame = capture.read()
                if not ok:
                    continue
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = _draw_exercise_pose(frame, result, exercise_code)
                filename = f"rep{repetition['number']}_{key_number}.jpg"
                cv2.imwrite(str(output_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                evidence_item = dict(item)
                evidence_item["image_url"] = f"/static/generated/{token}/{filename}"
                evidence.append(evidence_item)
            repetition["evidence"] = evidence
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
        adapted_row, model_input_adapter = adapt_model_input(
            candidate["row"], exercise, mode, info["feature_columns"]
        )
        frame = pd.DataFrame(
            [adapted_row],
            columns=info["feature_columns"],
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_probability = float(model.predict_proba(frame)[0, 1])
        probability = float(calibrated_probability(calibrator, [raw_probability])[0])
        status = prediction_label(probability, threshold)
        repetitions.append({
            "number": number,
            "status": status,
            "raw_probability": raw_probability,
            "probability": probability,
            "threshold": threshold,
            "margin": probability - threshold,
            "near_threshold": abs(probability - threshold) <= NEAR_THRESHOLD_MARGIN,
            "start": candidate.get("source_start", candidate["start"]),
            "end": candidate.get("source_end", candidate["end"]),
            "frames": candidate["used_frames"],
            "model_input_adapter": model_input_adapter,
        })

    generate_evidence_images(video_path, candidates, repetitions, exercise)

    median_probability = float(np.median([item["probability"] for item in repetitions]))
    final_status = prediction_label(median_probability, threshold)
    near_threshold = abs(median_probability - threshold) <= NEAR_THRESHOLD_MARGIN
    dominant_key = "dominant_arm" if exercise == "m07" else "dominant_leg"
    dominant_side = diagnostics.get(dominant_key) or "kedua sisi"
    feature_bases = sorted({re.split(r"_t\d{3}|__", column)[0] for column in info["feature_columns"]})
    upper_body_features = [name for name in feature_bases if name.startswith(("trunk", "shoulder"))]

    return {
        "video_name": original_name,
        "exercise_code": exercise,
        "exercise_name": EXERCISE_MAP[exercise],
        "status": final_status,
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
        "upper_body_features": upper_body_features,
        "visual_focus_text": VISUAL_FOCUS.get(
            exercise,
            "Garis hijau menekankan pinggul, lutut, dan kaki sebagai bagian utama gerakan.",
        ),
        "masked_features": diagnostics.get("masked_features", []),
        "input_adapter_version": diagnostics.get("input_adapter_version"),
        "input_adapter_features": diagnostics.get("input_adapter_features", []),
        "repetitions": repetitions,
        "warning": (
            "Hasil berada dekat ambang Kinect-only. Sebaiknya rekam ulang dari posisi kamera "
            "yang sama dan pastikan satu repetisi terlihat lengkap."
            if near_threshold else None
        ),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    selected_exercise = request.form.get("exercise", "m01")

    if request.method == "POST":
        try:
            if selected_exercise not in EXERCISE_MAP:
                raise ValueError("Jenis gerakan tidak valid.")
            video = request.files.get("video")
            suffix = validate_upload(video)
            safe_name = secure_filename(video.filename) or f"upload{suffix}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary_path = Path(temporary.name)
                video.save(temporary)
            try:
                result = analyze_video(temporary_path, safe_name, selected_exercise)
            finally:
                temporary_path.unlink(missing_ok=True)
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        exercises=EXERCISE_MAP,
        selected_exercise=selected_exercise,
        result=result,
        error=error,
    )


@app.errorhandler(413)
def upload_too_large(_error):
    return render_template(
        "index.html",
        exercises=EXERCISE_MAP,
        selected_exercise="m01",
        result=None,
        error="Ukuran video melebihi batas 250 MB.",
    ), 413


if __name__ == "__main__":
    port = int(os.environ.get("RF_APP_PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=False)
