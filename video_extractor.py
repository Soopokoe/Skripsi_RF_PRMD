from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from pose_pattern_utils import (
    angle_3d,
    body_aligned_normalization,
    extract_frame_features,
    frame_features_to_wide_pattern,
    mediapipe_result_to_points,
)


VISIBILITY_THRESHOLD = 0.65
DOMINANCE_MARGIN = 0.15
SIDE_VIEW_SHOULDER_RATIO = 0.25
INPUT_ADAPTER_VERSION = "mediapipe_to_uiprmd_v4"

# Fitur ini memerlukan kedua sisi tubuh. Pada tampak samping satu sisi, nilainya
# tidak boleh dipaksakan dari landmark yang tertutup; analyzer akan mengimputasi
# median training sehingga sudut tersebut dinilai dari fitur yang memang terlihat.
TWO_SIDE_FEATURES = {
    "hand_height_difference",
    "shoulder_height_difference",
    "hip_height_difference",
    "knee_height_difference",
    "ankle_height_difference",
    "foot_distance",
    "hand_distance",
    "ankle_lateral_distance",
    "ankle_sagittal_distance",
    "knee_lateral_distance",
    "knee_sagittal_distance",
}


# MediaPipe menghasilkan sudut anatomi dan kedalaman kamera monocular, sementara
# RF dilatih dari posisi global hasil forward-kinematics UI-PRMD. Adapter ini
# hanya bekerja pada input video. Model, probabilitas kalibrasi, dan threshold
# Kinect tidak disentuh.
MEDIAPIPE_TO_KINECT_CONFIG = {
    "m01": {
        "phase_equalization": "high_low_high",
        "flexion_gain": {"left_knee_angle": 1.20, "right_knee_angle": 1.20,
                         "left_hip_angle": 1.15, "right_hip_angle": 1.15},
        "vertical_gain": {"hip_to_ankle_height": 1.50},
        "trunk_baseline": 3.0,
        "trunk_gain": 0.65,
        "remove_baseline": ("hip_height_difference",),
    },
    "m04": {
        "phase_equalization": "high_low_high",
        "flexion_gain": {"left_knee_angle": 1.25, "right_knee_angle": 1.25},
        "trunk_baseline": 3.0,
        "trunk_gain": 0.55,
        "remove_baseline": ("hip_height_difference",),
        "minimum_ranges": {
            "foot_distance": 0.90,
            "ankle_lateral_distance": 0.80,
            "knee_lateral_distance": 0.62,
        },
        "canonical_active_knee": True,
    },
    "m05": {
        "flexion_gain": {"left_knee_angle": 2.10, "right_knee_angle": 2.10,
                         "left_hip_angle": 2.50, "right_hip_angle": 2.50},
        "vertical_gain": {"hip_to_ankle_height": 2.60},
        "trunk_baseline": 3.0,
        "trunk_gain": 1.60,
    },
}


def _finite_percentile(values, percentile, default=0.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if len(values) else float(default)


def _transform_feature(frame_features, name, transform):
    for item in frame_features:
        value = float(item.get(name, np.nan))
        if np.isfinite(value):
            item[name] = float(transform(value))


def _phase_equalize_low_hold(frame_features, exercise_code):
    """Normalize descent/hold/ascent duration while preserving the observed curve."""
    if len(frame_features) < 20:
        return frame_features
    left = np.asarray([item.get("left_knee_angle", np.nan) for item in frame_features], dtype=float)
    right = np.asarray([item.get("right_knee_angle", np.nan) for item in frame_features], dtype=float)
    signal = np.minimum(left, right) if exercise_code == "m04" else (left + right) / 2.0
    signal = _smooth(signal, window=7)
    low = _finite_percentile(signal, 10)
    high = _finite_percentile(signal, 90)
    if not np.isfinite(low + high) or high - low < 10.0:
        return frame_features
    plateau = np.where(signal <= low + 0.20 * (high - low))[0]
    if not len(plateau):
        return frame_features
    entry, exit_index = int(plateau[0]), int(plateau[-1])
    count = len(frame_features)
    if entry < 2 or exit_index >= count - 2 or exit_index <= entry:
        return frame_features

    descent_end = max(2, int(round((count - 1) * 0.25)))
    hold_end = min(count - 3, int(round((count - 1) * 0.75)))
    source_positions = np.concatenate([
        np.linspace(0, entry, descent_end + 1),
        np.linspace(entry, exit_index, hold_end - descent_end + 1)[1:],
        np.linspace(exit_index, count - 1, count - hold_end)[1:],
    ])
    source_positions = np.asarray(source_positions[:count], dtype=float)
    old_positions = np.arange(count, dtype=float)
    keys = sorted({key for item in frame_features for key in item})
    output = [dict() for _ in range(count)]
    for key in keys:
        values = np.asarray([item.get(key, np.nan) for item in frame_features], dtype=float)
        values = np.asarray(pd.Series(values).interpolate(limit_direction="both").bfill().ffill(), dtype=float)
        interpolated = np.interp(source_positions, old_positions, values)
        for index, value in enumerate(interpolated):
            output[index][key] = float(value)
    return output


def _stretch_about_high(frame_features, name, gain):
    values = [item.get(name, np.nan) for item in frame_features]
    anchor = _finite_percentile(values, 95)
    _transform_feature(frame_features, name, lambda value: anchor - gain * (anchor - value))


def _remove_perspective_baseline(frame_features, name):
    values = [item.get(name, np.nan) for item in frame_features]
    baseline = _finite_percentile(values, 5)
    _transform_feature(frame_features, name, lambda value: max(0.0, value - baseline))


def _normalize_trunk_curve(frame_features, baseline_target, gain):
    values = [item.get("trunk_angle", np.nan) for item in frame_features]
    source_baseline = _finite_percentile(values, 5)
    _transform_feature(
        frame_features,
        "trunk_angle",
        lambda value: max(0.0, baseline_target + gain * (value - source_baseline)),
    )


def _ensure_minimum_range(frame_features, name, target_range, maximum_gain=2.0):
    values = np.asarray([item.get(name, np.nan) for item in frame_features], dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return
    q05, q50, q95 = np.percentile(finite, [5, 50, 95])
    current_range = float(q95 - q05)
    if current_range < 1e-6 or current_range >= target_range:
        return
    gain = min(float(target_range / current_range), maximum_gain)
    _transform_feature(frame_features, name, lambda value: q50 + gain * (value - q50))


def _canonicalize_side_lunge_knee(frame_features):
    """Place the actively bending knee on the RF's canonical right side."""
    left = _finite_percentile([item.get("left_knee_angle", np.nan) for item in frame_features], 10, 180)
    right = _finite_percentile([item.get("right_knee_angle", np.nan) for item in frame_features], 10, 180)
    if left >= right:
        return
    for item in frame_features:
        item["left_knee_angle"], item["right_knee_angle"] = (
            item.get("right_knee_angle", np.nan),
            item.get("left_knee_angle", np.nan),
        )


def adapt_mediapipe_segment_to_kinect(frame_features, exercise_code):
    """Correct systematic sensor-domain bias without changing RF or threshold."""
    output = [dict(item) for item in frame_features]
    config = MEDIAPIPE_TO_KINECT_CONFIG.get(exercise_code)
    if not config or len(output) < 8:
        return output, []

    adapted = []
    if config.get("phase_equalization") == "high_low_high":
        output = _phase_equalize_low_hold(output, exercise_code)
        adapted.append("phase_duration")
    if config.get("canonical_active_knee"):
        _canonicalize_side_lunge_knee(output)
        adapted.append("canonical_active_knee")

    for name, gain in config.get("flexion_gain", {}).items():
        _transform_feature(
            output, name,
            lambda value, factor=gain: np.clip(180.0 - factor * (180.0 - value), 0.0, 180.0),
        )
        adapted.append(name)

    for name, gain in config.get("vertical_gain", {}).items():
        _stretch_about_high(output, name, gain)
        adapted.append(name)

    for name in config.get("remove_baseline", ()):
        _remove_perspective_baseline(output, name)
        adapted.append(name)

    if "trunk_gain" in config:
        _normalize_trunk_curve(
            output, config.get("trunk_baseline", 0.0), config["trunk_gain"]
        )
        adapted.append("trunk_angle")

    for name, target_range in config.get("minimum_ranges", {}).items():
        _ensure_minimum_range(output, name, target_range)
        adapted.append(name)

    return output, sorted(set(adapted))


def _smooth(values, window=7):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _movement_signal(frame_features, exercise_code):
    if exercise_code in {"m01", "m04", "m05"}:
        return np.asarray([
            (item["left_knee_angle"] + item["right_knee_angle"]) / 2.0
            for item in frame_features
        ])
    if exercise_code == "m07":
        return np.asarray([
            max(item.get("left_hand_height", item["hand_mean_height"]),
                item.get("right_hand_height", item["hand_mean_height"]))
            for item in frame_features
        ])
    return None


def _state_machine_segments(signal, pattern, minimum_length=10):
    signal = _smooth(signal)
    low = float(np.percentile(signal, 20))
    high = float(np.percentile(signal, 80))
    if not np.isfinite(low + high) or high - low < 1e-6:
        return []

    if pattern == "low_high":
        states = ("low", "high")
    elif pattern == "low_high_low":
        states = ("low", "high", "low")
    else:
        states = ("high", "low", "high")

    def reached(state, value):
        return value <= low if state == "low" else value >= high

    segments = []
    state_index = 0
    start = None
    for index, value in enumerate(signal):
        # Saat menunggu fase kedua, simpan frame terakhir yang masih berada pada
        # fase awal. Ini membuang jeda panjang sebelum gerakan benar-benar mulai.
        if state_index == 1 and reached(states[0], value):
            start = index
            continue
        if not reached(states[state_index], value):
            continue
        if state_index == 0:
            start = index
            state_index = 1
            continue
        state_index += 1
        if state_index == len(states):
            end = min(len(signal), index + 6)
            start_with_padding = max(0, int(start) - 5)
            if end - start_with_padding >= minimum_length:
                segments.append((start_with_padding, end))
            state_index = 0
            start = None
    return segments


def detect_repetition_segments(frame_features, exercise_code):
    signal = _movement_signal(frame_features, exercise_code)
    if signal is None or len(signal) < 20:
        return []
    pattern = {
        "m01": "high_low_high",
        "m04": "high_low_high",
        # UI-PRMD m05 adalah transisi duduk ke berdiri, bukan tiga repetisi penuh.
        "m05": "low_high",
        "m07": "low_high_low",
    }.get(exercise_code)
    return _state_machine_segments(signal, pattern) if pattern else []


def _keyframe_assessment(frame_feature, exercise_code, phase, context=None):
    """Return a transparent visual check; this is not a frame-level RF prediction."""
    context = context or {}
    left_knee = float(frame_feature.get("left_knee_angle", np.nan))
    right_knee = float(frame_feature.get("right_knee_angle", np.nan))
    bent_knee = min(left_knee, right_knee)
    straight_knee = max(left_knee, right_knee)

    metrics = {
        "Lutut kiri": f"{left_knee:.0f} deg",
        "Lutut kanan": f"{right_knee:.0f} deg",
    }

    if exercise_code == "m07":
        left_shoulder = float(frame_feature.get("left_shoulder_angle", np.nan))
        right_shoulder = float(frame_feature.get("right_shoulder_angle", np.nan))
        left_elbow = float(frame_feature.get("left_elbow_angle", np.nan))
        right_elbow = float(frame_feature.get("right_elbow_angle", np.nan))
        hand_height = float(frame_feature.get("hand_mean_height", np.nan))
        arm_metrics = {
            "Bahu kiri": f"{left_shoulder:.0f} deg",
            "Bahu kanan": f"{right_shoulder:.0f} deg",
            "Siku kiri": f"{left_elbow:.0f} deg",
            "Siku kanan": f"{right_elbow:.0f} deg",
            "Tinggi tangan": f"{hand_height:.2f}",
        }
        if phase == "Fase utama":
            left_good = left_shoulder >= 60 and left_elbow >= 120
            right_good = right_shoulder >= 60 and right_elbow >= 120
            good = left_good or right_good
            bilateral = left_good and right_good
            return {
                "tone": "good" if good else "review",
                "assessment": (
                    "Kedua lengan terangkat dan memanjang" if bilateral
                    else "Lengan aktif terangkat dan memanjang" if good
                    else "Posisi lengan perlu diperiksa"
                ),
                "detail": (
                    "Fokus utama Shoulder Abduction pada bahu, siku, dan tangan terlihat."
                    if good else "Sedikitnya satu bahu belum cukup terangkat atau siku masih menekuk."
                ),
                "metrics": arm_metrics,
            }
        lowered = max(left_shoulder, right_shoulder) <= 55
        return {
            "tone": "good" if lowered else "neutral",
            "assessment": "Posisi lengan awal/akhir terlihat" if lowered else "Transisi lengan terdeteksi",
            "detail": (
                "Kedua lengan berada dekat posisi turun."
                if lowered else "Frame ini masih berada pada transisi gerakan lengan."
            ),
            "metrics": arm_metrics,
        }

    if phase in {"Awal", "Akhir"} and exercise_code in {"m01", "m04"}:
        good = left_knee >= 145 and right_knee >= 145
        return {
            "tone": "good" if good else "review",
            "assessment": "Posisi tegak/akhir terlihat" if good else "Posisi awal/akhir belum sepenuhnya tegak",
            "detail": "Kedua lutut mendekati lurus." if good else "Sedikitnya satu lutut masih cukup menekuk.",
            "metrics": metrics,
        }

    if phase == "Fase utama" and exercise_code == "m01":
        average = (left_knee + right_knee) / 2.0
        symmetric = abs(left_knee - right_knee) <= 20
        good = average <= 125 and symmetric
        return {
            "tone": "good" if good else "review",
            "assessment": "Tekukan squat terlihat seimbang" if good else "Tekukan squat perlu diperiksa",
            "detail": (
                "Kedua lutut menekuk dengan kedalaman yang serupa."
                if good else "Kedalaman atau keseimbangan tekukan lutut belum kuat."
            ),
            "metrics": metrics,
        }

    if phase == "Fase utama" and exercise_code == "m04":
        good = bent_knee <= 125 and straight_knee >= 145
        return {
            "tone": "good" if good else "review",
            "assessment": "Pola satu lutut tekuk dan satu lurus terlihat" if good else "Pemisahan tekuk-lurus perlu diperiksa",
            "detail": (
                "Indikator visual dasar Side Lunge terpenuhi."
                if good else "Perbedaan sudut kedua lutut belum cukup jelas."
            ),
            "metrics": metrics,
        }

    return {
        "tone": "neutral",
        "assessment": "Fase gerakan berhasil ditemukan",
        "detail": "Keputusan Correct/Incorrect tetap dihitung dari pola satu repetisi penuh.",
        "metrics": metrics,
    }


def _candidate_keyframes(frame_features, source_frame_indexes, start, end, exercise_code):
    if end <= start:
        return []
    indexes = list(range(start, end))
    segment = frame_features[start:end]
    endpoint_features = [segment[0], segment[-1]]
    def finite_median(name):
        values = np.asarray([item.get(name, np.nan) for item in endpoint_features], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.median(values)) if len(values) else 0.0

    context = {
        "ankle_baseline": finite_median("ankle_height_difference"),
        "knee_baseline": finite_median("knee_height_difference"),
        "hand_baseline": finite_median("hand_mean_height"),
    }
    signal = _movement_signal(segment, exercise_code)
    if signal is None or len(signal) == 0:
        main_index = indexes[len(indexes) // 2]
    elif exercise_code in {"m01", "m04", "m05"}:
        main_index = start + int(np.nanargmin(signal))
    else:
        main_index = start + int(np.nanargmax(signal))

    requested = [("Awal", start), ("Fase utama", main_index), ("Akhir", end - 1)]
    evidence = []
    used = set()
    for phase, detected_index in requested:
        detected_index = int(np.clip(detected_index, 0, len(frame_features) - 1))
        source_index = int(source_frame_indexes[detected_index])
        if source_index in used:
            continue
        used.add(source_index)
        item = {
            "phase": phase,
            "source_frame": source_index,
            "detected_frame": detected_index,
        }
        item.update(_keyframe_assessment(
            frame_features[detected_index], exercise_code, phase, context=context
        ))
        evidence.append(item)
    return evidence


def _score(visibility, names):
    return float(min(visibility.get(name, 0.0) for name in names))


def _copy_visible_side(features, aligned, visibility, is_side_view):
    left_leg = _score(visibility, ("left_hip", "left_knee", "left_ankle"))
    right_leg = _score(visibility, ("right_hip", "right_knee", "right_ankle"))
    left_arm = _score(visibility, ("left_shoulder", "left_elbow", "left_wrist"))
    right_arm = _score(visibility, ("right_shoulder", "right_elbow", "right_wrist"))

    dominant_leg = None
    if is_side_view or abs(left_leg - right_leg) >= DOMINANCE_MARGIN:
        dominant_leg = "left" if left_leg >= right_leg else "right"
        hidden_leg = "right" if dominant_leg == "left" else "left"
        features[f"{hidden_leg}_knee_angle"] = features[f"{dominant_leg}_knee_angle"]
        features[f"{hidden_leg}_hip_angle"] = features[f"{dominant_leg}_hip_angle"]
        features["hip_to_ankle_height"] = (
            aligned[f"{dominant_leg}_hip"][1] - aligned[f"{dominant_leg}_ankle"][1]
        )
        features["knee_mean_height"] = aligned[f"{dominant_leg}_knee"][1]

    dominant_arm = None
    if is_side_view or abs(left_arm - right_arm) >= DOMINANCE_MARGIN:
        dominant_arm = "left" if left_arm >= right_arm else "right"
        hidden_arm = "right" if dominant_arm == "left" else "left"
        features[f"{hidden_arm}_shoulder_angle"] = features[f"{dominant_arm}_shoulder_angle"]
        features[f"{hidden_arm}_elbow_angle"] = features[f"{dominant_arm}_elbow_angle"]
        hand = aligned.get(f"{dominant_arm}_hand", aligned[f"{dominant_arm}_wrist"])
        features["hand_mean_height"] = hand[1]

    dominant_body = dominant_leg or dominant_arm
    if is_side_view and dominant_body:
        shoulder = aligned[f"{dominant_body}_shoulder"]
        hip = aligned[f"{dominant_body}_hip"]
        features["trunk_angle"] = 180.0 - angle_3d(
            shoulder + np.array([0.0, 1.0, 0.0]), shoulder, hip
        )
        for feature in TWO_SIDE_FEATURES:
            features[feature] = np.nan

    return {
        "left_leg": left_leg,
        "right_leg": right_leg,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "dominant_leg": dominant_leg,
        "dominant_arm": dominant_arm,
    }


def _visibility_map(result, pose_landmark):
    landmark = result.pose_landmarks.landmark
    mapping = {
        "left_shoulder": pose_landmark.LEFT_SHOULDER,
        "right_shoulder": pose_landmark.RIGHT_SHOULDER,
        "left_elbow": pose_landmark.LEFT_ELBOW,
        "right_elbow": pose_landmark.RIGHT_ELBOW,
        "left_wrist": pose_landmark.LEFT_WRIST,
        "right_wrist": pose_landmark.RIGHT_WRIST,
        "left_hip": pose_landmark.LEFT_HIP,
        "right_hip": pose_landmark.RIGHT_HIP,
        "left_knee": pose_landmark.LEFT_KNEE,
        "right_knee": pose_landmark.RIGHT_KNEE,
        "left_ankle": pose_landmark.LEFT_ANKLE,
        "right_ankle": pose_landmark.RIGHT_ANKLE,
    }
    return {name: float(landmark[index].visibility) for name, index in mapping.items()}


def _shoulder_separation_ratio(result, pose_landmark):
    landmark = result.pose_landmarks.landmark
    left_shoulder = landmark[pose_landmark.LEFT_SHOULDER]
    right_shoulder = landmark[pose_landmark.RIGHT_SHOULDER]
    left_hip = landmark[pose_landmark.LEFT_HIP]
    right_hip = landmark[pose_landmark.RIGHT_HIP]
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    hip_y = (left_hip.y + right_hip.y) / 2.0
    torso = abs(shoulder_y - hip_y)
    return abs(left_shoulder.x - right_shoulder.x) / max(torso, 1e-6)


def _mask_unobservable_row_features(row, feature_names):
    for feature in feature_names:
        if feature not in TWO_SIDE_FEATURES:
            continue
        for key in list(row):
            if key.startswith(feature + "_t") or key.startswith(feature + "__"):
                row[key] = np.nan


def video_to_visibility_aware_pattern(video_path, exercise_code, target_steps=80):
    video_path = Path(video_path)
    pose_module = mp.solutions.pose
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka: {video_path}")
    # Rekaman ponsel sering menyimpan frame landscape dan metadata rotasi 90°.
    # Tanpa auto-orientation, fitur vertikal (tinggi tangan/pinggul dan torso)
    # dihitung pada orang yang tampak menyamping.
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_features = []
    view_flags = []
    visibility_rows = []
    dominant_legs = []
    dominant_arms = []
    source_frame_indexes = []

    with pose_module.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        source_frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_source_frame = source_frame_index
            source_frame_index += 1
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                continue

            points = mediapipe_result_to_points(result, pose_module)
            aligned = body_aligned_normalization(points)
            if aligned is None:
                continue

            visibility = _visibility_map(result, pose_module.PoseLandmark)
            shoulder_ratio = _shoulder_separation_ratio(result, pose_module.PoseLandmark)
            left_leg = _score(visibility, ("left_hip", "left_knee", "left_ankle"))
            right_leg = _score(visibility, ("right_hip", "right_knee", "right_ankle"))
            side_view = (
                shoulder_ratio < SIDE_VIEW_SHOULDER_RATIO
                or min(left_leg, right_leg) < VISIBILITY_THRESHOLD <= max(left_leg, right_leg)
            )

            features = extract_frame_features(aligned)
            side_info = _copy_visible_side(features, aligned, visibility, side_view)
            frame_features.append(features)
            source_frame_indexes.append(current_source_frame)
            view_flags.append(side_view)
            visibility_rows.append(side_info)
            if side_info["dominant_leg"]:
                dominant_legs.append(side_info["dominant_leg"])
            if side_info["dominant_arm"]:
                dominant_arms.append(side_info["dominant_arm"])

    capture.release()
    if len(frame_features) < 8:
        raise RuntimeError("Landmark tubuh terlalu sedikit terbaca.")

    repetition_segments = detect_repetition_segments(frame_features, exercise_code)
    segments_to_score = repetition_segments or [(0, len(frame_features))]
    candidates = []
    for repetition_start, repetition_end in segments_to_score:
        adapted_segment, adapted_features = adapt_mediapipe_segment_to_kinect(
            frame_features[repetition_start:repetition_end], exercise_code
        )
        candidate_row, candidate_features, candidate_start, candidate_end, candidate_frames = (
            frame_features_to_wide_pattern(
                adapted_segment,
                exercise_code,
                target_steps=target_steps,
                # State-machine sudah memotong satu repetisi. Trimming kedua
                # dapat membuang fase awal pada repetisi pendek Side Lunge.
                trim=(not bool(repetition_segments) or exercise_code == "m07"),
            )
        )
        absolute_start = candidate_start + repetition_start
        absolute_end = candidate_end + repetition_start
        candidates.append({
            "row": candidate_row,
            "feature_names": candidate_features,
            "start": absolute_start,
            "end": absolute_end,
            "used_frames": candidate_frames,
            "source_segment": [repetition_start, repetition_end],
            "source_start": source_frame_indexes[absolute_start],
            "source_end": source_frame_indexes[max(absolute_start, absolute_end - 1)],
            "input_adapter_features": adapted_features,
            "keyframes": _candidate_keyframes(
                frame_features, source_frame_indexes, absolute_start, absolute_end, exercise_code
            ),
        })

    selected_repetition = segments_to_score[0]
    row = candidates[0]["row"]
    feature_names = candidates[0]["feature_names"]
    start = candidates[0]["start"]
    end = candidates[0]["end"]
    used_frames = candidates[0]["used_frames"]
    one_side_ratio = float(np.mean(view_flags)) if view_flags else 0.0
    masked_features = []
    if one_side_ratio >= 0.5:
        masked_features = sorted(set(feature_names) & TWO_SIDE_FEATURES)
        for candidate in candidates:
            _mask_unobservable_row_features(candidate["row"], masked_features)

    def most_common(values):
        if not values:
            return None
        return max(set(values), key=values.count)

    diagnostics = {
        "view_mode": "one_side" if one_side_ratio >= 0.5 else "multi_side",
        "one_side_frame_ratio": round(one_side_ratio, 4),
        "dominant_leg": most_common(dominant_legs),
        "dominant_arm": most_common(dominant_arms),
        "masked_features": masked_features,
        "detected_frames": len(frame_features),
        "total_frames": total_frames,
        "pose_detection_ratio": round(len(frame_features) / max(total_frames, 1), 4),
        "detected_repetitions": len(repetition_segments),
        "repetition_segments": [list(segment) for segment in repetition_segments],
        "selected_repetition": list(selected_repetition),
        "mean_visibility": {
            name: round(float(np.mean([item[name] for item in visibility_rows])), 4)
            for name in ("left_leg", "right_leg", "left_arm", "right_arm")
        },
        "input_adapter_version": INPUT_ADAPTER_VERSION,
        "input_adapter_features": sorted({
            name for candidate in candidates
            for name in candidate.get("input_adapter_features", [])
        }),
    }
    return row, feature_names, start, end, used_frames, diagnostics, candidates
