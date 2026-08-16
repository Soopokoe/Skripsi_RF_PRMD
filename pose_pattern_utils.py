import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except Exception:
    mp = None


EXERCISE_MAP = {
    "m01": "Deep Squat",
    "m04": "Side Lunge",
    "m05": "Sit to Stand",
    "m07": "Standing Shoulder Abduction",
}

SELECTED_EXERCISES_DEFAULT = set(EXERCISE_MAP.keys())

# Urutan resmi 22 joint Kinect pada UI-PRMD (Table 2 dataset paper):
# waist, spine, chest, neck, head, head-tip, lalu rantai lengan dan tungkai.
# Hanya waist yang absolut; joint lain relatif terhadap parent pada skeleton.
JOINT_NAMES = [
    "spine_base", "spine_mid", "chest", "neck", "head", "head_tip",
    "left_collar", "left_shoulder", "left_elbow", "left_hand",
    "right_collar", "right_shoulder", "right_elbow", "right_hand",
    "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
]
JOINT = {name: index for index, name in enumerate(JOINT_NAMES)}
# UI-PRMD tidak menyediakan joint wrist terpisah. Posisi hand dipakai sebagai
# padanan landmark wrist MediaPipe untuk perhitungan sudut siku dan tangan.
JOINT["left_wrist"] = JOINT["left_hand"]
JOINT["right_wrist"] = JOINT["right_hand"]

LEFT_RIGHT_JOINT_PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_hand", "right_hand"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_foot", "right_foot"),
]

# Rotasi yaw sintetis hilang kembali setelah body-aligned normalization dan
# hanya menghasilkan duplikat berbobot besar, bukan variasi informasi baru.
CAMERA_ROTATION_DEGREES = [0]
SIDE_VIEW_AUGMENT_EXERCISES = set()
FLIP_AUGMENT_EXERCISES = {"m04", "m07"}

FEATURES_BY_EXERCISE = {
    "m01": [
        "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle",
        "hip_to_ankle_height", "trunk_angle", "shoulder_height_difference", "hip_height_difference",
    ],
    "m04": [
        "foot_distance", "ankle_lateral_distance", "knee_lateral_distance",
        "left_knee_angle", "right_knee_angle", "hip_height_difference", "trunk_angle",
    ],
    "m05": [
        "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle",
        "hip_to_ankle_height", "trunk_angle", "shoulder_height_difference",
    ],
    "m07": [
        "dominant_shoulder_angle", "dominant_elbow_angle", "dominant_hand_height",
        "trunk_angle", "shoulder_height_difference",
    ],
}

FEEDBACK_MAP = {
    "m01": {
        "left_knee_angle": "Pola tekukan lutut belum mengikuti pola Deep Squat yang benar.",
        "right_knee_angle": "Pola tekukan lutut belum mengikuti pola Deep Squat yang benar.",
        "left_hip_angle": "Pola gerak pinggul belum sesuai Deep Squat.",
        "right_hip_angle": "Pola gerak pinggul belum sesuai Deep Squat.",
        "hip_to_ankle_height": "Pola turun-naik tubuh belum sesuai gerakan Deep Squat yang benar.",
        "trunk_angle": "Pola posisi badan tidak stabil saat Deep Squat.",
        "shoulder_height_difference": "Bahu kurang stabil selama pola Deep Squat.",
        "hip_height_difference": "Pinggul kurang stabil selama pola Deep Squat.",
    },
    "m04": {
        "foot_distance": "Pola buka-tutup kaki ke samping belum sesuai Side Lunge.",
        "ankle_lateral_distance": "Arah gerakan kaki ke samping belum mengikuti pola Side Lunge.",
        "knee_lateral_distance": "Pola posisi lutut ke samping belum sesuai Side Lunge.",
        "left_knee_angle": "Pola tekukan lutut tumpuan belum sesuai Side Lunge.",
        "right_knee_angle": "Pola tekukan lutut tumpuan belum sesuai Side Lunge.",
        "hip_height_difference": "Pinggul kurang stabil selama Side Lunge.",
        "trunk_angle": "Badan terlalu menyimpang dari pola Side Lunge.",
    },
    "m05": {
        "left_knee_angle": "Pola tekukan lutut saat duduk-berdiri belum sesuai.",
        "right_knee_angle": "Pola tekukan lutut saat duduk-berdiri belum sesuai.",
        "left_hip_angle": "Pola gerak pinggul saat duduk-berdiri belum sesuai.",
        "right_hip_angle": "Pola gerak pinggul saat duduk-berdiri belum sesuai.",
        "hip_to_ankle_height": "Pola perubahan tinggi tubuh belum sesuai Sit to Stand.",
        "trunk_angle": "Badan kurang stabil selama Sit to Stand.",
        "shoulder_height_difference": "Bahu kurang stabil selama Sit to Stand.",
    },
    "m07": {
        "dominant_shoulder_angle": "Pola angkatan bahu pada lengan aktif belum sesuai Shoulder Abduction.",
        "dominant_elbow_angle": "Pola siku pada lengan aktif belum sesuai saat mengangkat lengan.",
        "dominant_hand_height": "Pola kenaikan tangan aktif belum sesuai Shoulder Abduction.",
        "trunk_angle": "Badan kurang stabil saat mengangkat lengan.",
        "shoulder_height_difference": "Bahu kurang stabil saat mengangkat lengan.",
    },
}


def parse_selected_exercises(value):
    if value is None or value.strip() == "":
        return SELECTED_EXERCISES_DEFAULT
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def infer_label(path: Path):
    text = str(path).lower()
    if "incorrect" in text:
        return 0, "Incorrect"
    if "correct" in text:
        return 1, "Correct"
    return None, None


def infer_exercise(path: Path):
    text = str(path).lower()
    match = re.search(r"m\d{2}", text)
    if not match:
        return None, None
    code = match.group(0)
    return code, EXERCISE_MAP.get(code, code)


def infer_subject(path: Path):
    text = str(path).lower()
    match = re.search(r"s\d{2}", text)
    return match.group(0) if match else "unknown"


def infer_repetition(path: Path):
    text = str(path).lower()
    match = re.search(r"e\d{2}", text)
    return match.group(0) if match else "unknown"


def safe_stem(path: Path):
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", path.stem)


def read_position_file(path: Path):
    df = pd.read_csv(path, header=None, sep=r"[\s,]+", engine="python")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    return df


def row_to_joints(row):
    values = np.asarray(row, dtype=float)
    usable_length = (len(values) // 3) * 3
    if usable_length < 60:
        return None
    return values[:usable_length].reshape(-1, 3)


UIPRMD_PARENTS = (
    None, 0, 1, 2, 3, 4,       # waist -> head tip
    2, 6, 7, 8,                # left arm from chest
    2, 10, 11, 12,             # right arm from chest
    0, 14, 15, 16,             # left leg from waist
    0, 18, 19, 20,             # right leg from waist
)


def paired_angle_path(position_path: Path):
    """Return the UI-PRMD Angles file paired with a Positions file."""
    position_path = Path(position_path)
    parts = list(position_path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "positions":
            parts[index] = "Angles"
            break
    angle_path = Path(*parts)
    return angle_path.with_name(angle_path.name.replace("_positions", "_angles"))


def _rotation_x(theta):
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=float)


def _rotation_y(theta):
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=float)


def _rotation_z(theta):
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=float)


def uiprmd_euler_to_rotation(angle_triplet):
    """Convert a recorded UI-PRMD Euler triplet using its reference convention."""
    first, second, third = np.deg2rad(np.asarray(angle_triplet, dtype=float))
    # Sesuai visualizer referensi UI-PRMD: Rz(third) Ry(second) Rx(first).
    return _rotation_z(third) @ _rotation_y(second) @ _rotation_x(first)


def reconstruct_kinect_positions(joints, angles=None):
    """Reconstruct global UI-PRMD joints with hierarchical forward kinematics."""
    joints = np.asarray(joints, dtype=float)
    if angles is None:
        raise ValueError("Rekonstruksi UI-PRMD memerlukan file Angles pasangan.")
    angles = np.asarray(angles, dtype=float)
    n = min(len(joints), len(angles), len(UIPRMD_PARENTS))
    world = np.zeros_like(joints, dtype=float)
    world_rotation = [None] * n
    world[0] = joints[0]
    world_rotation[0] = uiprmd_euler_to_rotation(angles[0])

    for child in range(1, n):
        parent = UIPRMD_PARENTS[child]
        parent_rotation = world_rotation[parent]
        world[child] = world[parent] + parent_rotation @ joints[child]
        world_rotation[child] = parent_rotation @ uiprmd_euler_to_rotation(angles[child])
    return world


def rotate_y_joints(joints, degrees):
    theta = np.deg2rad(degrees)
    rotation_matrix = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)],
    ], dtype=float)
    return joints @ rotation_matrix.T


def flip_lr_joints(joints):
    output = joints.copy()
    output[:, 0] = -output[:, 0]

    for left_name, right_name in LEFT_RIGHT_JOINT_PAIRS:
        left_index = JOINT[left_name]
        right_index = JOINT[right_name]
        if left_index < len(output) and right_index < len(output):
            output[[left_index, right_index]] = output[[right_index, left_index]]

    return output


def build_augmentation_configs(exercise_code):
    # Untuk classifier dua kelas, Correct dan Incorrect diaugmentasi sama supaya model adil.
    base_augmentations = ["original"]

    if exercise_code in FLIP_AUGMENT_EXERCISES:
        base_augmentations.append("flip_lr")

    if exercise_code in SIDE_VIEW_AUGMENT_EXERCISES:
        base_augmentations.extend(["side_right_90", "side_left_90"])

    configs = []
    for base_aug in base_augmentations:
        for rotation in CAMERA_ROTATION_DEGREES:
            if rotation == 0:
                rot_name = "rot_0"
            elif rotation > 0:
                rot_name = f"rot_right_{rotation}"
            else:
                rot_name = f"rot_left_{abs(rotation)}"

            configs.append({
                "augmentation": f"{base_aug}__{rot_name}",
                "base_augmentation": base_aug,
                "camera_rotation_degree": rotation,
            })
    return configs


def apply_augmentation(joints, config):
    output = joints.copy()
    base_aug = config["base_augmentation"]
    rotation = config["camera_rotation_degree"]

    if base_aug == "flip_lr":
        output = flip_lr_joints(output)
    elif base_aug == "side_right_90":
        output = rotate_y_joints(output, 90)
    elif base_aug == "side_left_90":
        output = rotate_y_joints(output, -90)

    if rotation != 0:
        output = rotate_y_joints(output, rotation)

    return output


def safe_unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=float)
    return vector / norm


def angle_3d(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return np.nan
    cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def extract_named_points_from_kinect_joints(joints):
    points = {}
    for name, index in JOINT.items():
        if index < len(joints):
            points[name] = np.asarray(joints[index], dtype=float)

    required = [
        "left_hip", "right_hip", "left_shoulder", "right_shoulder",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    ]
    for name in required:
        if name not in points:
            return None
    return points


def body_aligned_normalization(points):
    left_hip = points["left_hip"]
    right_hip = points["right_hip"]
    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]

    mid_hip = (left_hip + right_hip) / 2.0
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0

    # Kinect menyimpan Y sebagai sumbu vertikal. Jangan memakai arah torso
    # sebagai up-axis: cara lama selalu meluruskan torso pada setiap frame dan
    # menghapus trunk lean serta ketidaksejajaran pinggul/bahu yang justru
    # merupakan sinyal Correct vs Incorrect.
    up_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    right_axis = right_hip - left_hip
    right_axis = right_axis - np.dot(right_axis, up_axis) * up_axis
    right_axis = safe_unit(right_axis)

    if np.linalg.norm(right_axis) == 0:
        right_axis = right_shoulder - left_shoulder
        right_axis = right_axis - np.dot(right_axis, up_axis) * up_axis
        right_axis = safe_unit(right_axis)

    forward_axis = safe_unit(np.cross(right_axis, up_axis))
    if np.linalg.norm(forward_axis) == 0:
        return None

    right_axis = safe_unit(np.cross(up_axis, forward_axis))

    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    hip_width = np.linalg.norm(right_hip - left_hip)
    torso_length = np.linalg.norm(mid_shoulder - mid_hip)
    scale = max(shoulder_width, hip_width, torso_length, 1e-6)

    aligned = {}
    for key, point in points.items():
        centered = point - mid_hip
        aligned[key] = np.array([
            np.dot(centered, right_axis) / scale,
            np.dot(centered, up_axis) / scale,
            np.dot(centered, forward_axis) / scale,
        ], dtype=float)
    return aligned


def extract_frame_features(points):
    p = points

    mid_hip = (p["left_hip"] + p["right_hip"]) / 2.0
    mid_shoulder = (p["left_shoulder"] + p["right_shoulder"]) / 2.0
    mid_ankle = (p["left_ankle"] + p["right_ankle"]) / 2.0

    left_hand = p.get("left_hand", p["left_wrist"])
    right_hand = p.get("right_hand", p["right_wrist"])
    left_foot = p.get("left_foot", p["left_ankle"])
    right_foot = p.get("right_foot", p["right_ankle"])

    return {
        "left_knee_angle": angle_3d(p["left_hip"], p["left_knee"], p["left_ankle"]),
        "right_knee_angle": angle_3d(p["right_hip"], p["right_knee"], p["right_ankle"]),
        "left_hip_angle": angle_3d(p["left_shoulder"], p["left_hip"], p["left_knee"]),
        "right_hip_angle": angle_3d(p["right_shoulder"], p["right_hip"], p["right_knee"]),
        "left_elbow_angle": angle_3d(p["left_shoulder"], p["left_elbow"], p["left_wrist"]),
        "right_elbow_angle": angle_3d(p["right_shoulder"], p["right_elbow"], p["right_wrist"]),
        "left_shoulder_angle": angle_3d(p["left_elbow"], p["left_shoulder"], p["left_hip"]),
        "right_shoulder_angle": angle_3d(p["right_elbow"], p["right_shoulder"], p["right_hip"]),
        "trunk_angle": 180.0 - angle_3d(
            mid_shoulder + np.array([0, 1, 0]), mid_shoulder, mid_hip
        ),

        "hand_height_difference": abs(left_hand[1] - right_hand[1]),
        "shoulder_height_difference": abs(p["left_shoulder"][1] - p["right_shoulder"][1]),
        "hip_height_difference": abs(p["left_hip"][1] - p["right_hip"][1]),
        "knee_height_difference": abs(p["left_knee"][1] - p["right_knee"][1]),
        "ankle_height_difference": abs(p["left_ankle"][1] - p["right_ankle"][1]),

        "foot_distance": dist(left_foot, right_foot),
        "hand_distance": dist(left_hand, right_hand),
        "ankle_lateral_distance": abs(p["left_ankle"][0] - p["right_ankle"][0]),
        "ankle_sagittal_distance": abs(p["left_ankle"][2] - p["right_ankle"][2]),
        "knee_lateral_distance": abs(p["left_knee"][0] - p["right_knee"][0]),
        "knee_sagittal_distance": abs(p["left_knee"][2] - p["right_knee"][2]),

        "hip_to_ankle_height": mid_hip[1] - mid_ankle[1],
        "left_hand_height": left_hand[1],
        "right_hand_height": right_hand[1],
        "hand_mean_height": (left_hand[1] + right_hand[1]) / 2.0,
        "knee_mean_height": (p["left_knee"][1] + p["right_knee"][1]) / 2.0,
    }


def trim_main_segment(frame_features, exercise_code):
    if len(frame_features) < 20:
        return frame_features, 0, len(frame_features)

    df = pd.DataFrame(frame_features)
    keys = FEATURES_BY_EXERCISE.get(exercise_code, list(df.columns))
    keys = [key for key in keys if key in df.columns]
    if not keys:
        return frame_features, 0, len(frame_features)

    motion = df[keys].diff().abs().sum(axis=1).fillna(0).to_numpy()
    if len(motion) >= 7:
        kernel = np.ones(7) / 7
        motion = np.convolve(motion, kernel, mode="same")

    threshold = np.percentile(motion, 60)
    active = np.where(motion >= threshold)[0]
    if len(active) == 0:
        return frame_features, 0, len(frame_features)

    start = max(0, int(active[0]) - 8)
    end = min(len(frame_features), int(active[-1]) + 9)
    if end - start < 12:
        return frame_features, 0, len(frame_features)
    return frame_features[start:end], start, end


def resample_curve(values, target_steps):
    values = pd.Series(values).astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.interpolate(limit_direction="both").bfill().ffill().fillna(0).to_numpy()

    if len(values) == 0:
        return np.zeros(target_steps)
    if len(values) == 1:
        return np.repeat(values[0], target_steps)

    old_x = np.linspace(0, 1, len(values))
    new_x = np.linspace(0, 1, target_steps)
    return np.interp(new_x, old_x, values)


def normalize_curve_shape(values, clip_value=3.0):
    values = np.asarray(values, dtype=float)
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan)
    values = values.interpolate(limit_direction="both").bfill().ffill().fillna(0).to_numpy()

    median = np.nanmedian(values)
    q05 = np.nanpercentile(values, 5)
    q95 = np.nanpercentile(values, 95)
    scale = q95 - q05

    if not np.isfinite(scale) or abs(scale) < 1e-6:
        scale = np.nanstd(values)

    if not np.isfinite(scale) or abs(scale) < 1e-6:
        return np.zeros_like(values, dtype=float)

    normalized = (values - median) / scale
    return np.clip(normalized, -clip_value, clip_value)


def _add_exercise_derived_features(frame_features, exercise_code):
    output = [dict(item) for item in frame_features]
    if exercise_code != "m07" or not output:
        return output

    left_motion = np.nanpercentile([item["left_shoulder_angle"] for item in output], 95) - np.nanpercentile(
        [item["left_shoulder_angle"] for item in output], 5
    )
    right_motion = np.nanpercentile([item["right_shoulder_angle"] for item in output], 95) - np.nanpercentile(
        [item["right_shoulder_angle"] for item in output], 5
    )
    dominant = "left" if left_motion >= right_motion else "right"
    for item in output:
        item["dominant_shoulder_angle"] = item[f"{dominant}_shoulder_angle"]
        item["dominant_elbow_angle"] = item[f"{dominant}_elbow_angle"]
        item["dominant_hand_height"] = item[f"{dominant}_hand_height"]
    return output


def frame_features_to_wide_pattern(frame_features, exercise_code, target_steps=80, trim=True):
    frame_features = _add_exercise_derived_features(frame_features, exercise_code)
    if trim:
        trimmed, start, end = trim_main_segment(frame_features, exercise_code)
    else:
        trimmed = frame_features
        start, end = 0, len(frame_features)
    if len(trimmed) < 8:
        trimmed = frame_features
        start, end = 0, len(frame_features)

    df = pd.DataFrame(trimmed)
    features = FEATURES_BY_EXERCISE.get(exercise_code, list(df.columns))
    row = {}

    for feature in features:
        if feature not in df.columns:
            raw_curve = np.zeros(target_steps)
        else:
            raw_curve = resample_curve(df[feature].to_numpy(), target_steps)

        # Pola yang dinormalisasi membuat model tahan terhadap perbedaan skala
        # antar-orang/kamera, tetapi normalisasi per-sekuens juga menghilangkan
        # amplitudo absolut (mis. kedalaman tekukan lutut). Simpan keduanya:
        # kurva shape untuk kompatibilitas dan statistik raw untuk informasi
        # besar gerakan tanpa menggandakan ratusan kolom time-series.
        curve = normalize_curve_shape(raw_curve)

        for t in range(target_steps):
            row[f"{feature}_t{t:03d}"] = float(curve[t])

        raw_curve = np.asarray(raw_curve, dtype=float)
        velocity = np.diff(raw_curve)
        row[f"{feature}__raw_median"] = float(np.median(raw_curve))
        row[f"{feature}__raw_q05"] = float(np.percentile(raw_curve, 5))
        row[f"{feature}__raw_q95"] = float(np.percentile(raw_curve, 95))
        row[f"{feature}__raw_range"] = float(np.percentile(raw_curve, 95) - np.percentile(raw_curve, 5))
        row[f"{feature}__raw_std"] = float(np.std(raw_curve))
        row[f"{feature}__raw_start"] = float(raw_curve[0])
        row[f"{feature}__raw_end"] = float(raw_curve[-1])
        row[f"{feature}__peak_time"] = float(np.argmax(raw_curve) / max(target_steps - 1, 1))
        row[f"{feature}__valley_time"] = float(np.argmin(raw_curve) / max(target_steps - 1, 1))
        row[f"{feature}__mean_abs_velocity"] = float(np.mean(np.abs(velocity))) if len(velocity) else 0.0

    return row, features, start, end, len(trimmed)


def feature_deviation_from_templates(row, features, correct_template, incorrect_template=None, target_steps=80):
    deviations = []
    for feature in features:
        user_curve = np.array([row.get(f"{feature}_t{t:03d}", 0.0) for t in range(target_steps)], dtype=float)
        correct_curve = np.array(correct_template.get(feature, [0.0] * target_steps), dtype=float)

        rmse_correct = float(np.sqrt(np.mean((user_curve - correct_curve) ** 2)))

        item = {
            "feature": feature,
            "rmse_correct": round(rmse_correct, 4),
            "score": round(rmse_correct, 4),
        }

        if incorrect_template is not None and feature in incorrect_template:
            incorrect_curve = np.array(incorrect_template.get(feature, [0.0] * target_steps), dtype=float)
            rmse_incorrect = float(np.sqrt(np.mean((user_curve - incorrect_curve) ** 2)))
            item["rmse_incorrect"] = round(rmse_incorrect, 4)

        deviations.append(item)

    return sorted(deviations, key=lambda x: x["score"], reverse=True)


def mediapipe_result_to_points(result, mp_pose):
    world = result.pose_world_landmarks is not None
    landmarks = result.pose_world_landmarks.landmark if world else result.pose_landmarks.landmark

    def point(landmark_id):
        lm = landmarks[landmark_id]
        if world:
            # MediaPipe memakai Y ke bawah, sedangkan pipeline Kinect dan
            # body_aligned_normalization memakai Y ke atas.
            return np.array([lm.x, -lm.y, lm.z], dtype=float)
        return np.array([lm.x, -lm.y, lm.z], dtype=float)

    points = {
        "left_shoulder": point(mp_pose.PoseLandmark.LEFT_SHOULDER),
        "right_shoulder": point(mp_pose.PoseLandmark.RIGHT_SHOULDER),
        "left_elbow": point(mp_pose.PoseLandmark.LEFT_ELBOW),
        "right_elbow": point(mp_pose.PoseLandmark.RIGHT_ELBOW),
        "left_wrist": point(mp_pose.PoseLandmark.LEFT_WRIST),
        "right_wrist": point(mp_pose.PoseLandmark.RIGHT_WRIST),
        "left_hip": point(mp_pose.PoseLandmark.LEFT_HIP),
        "right_hip": point(mp_pose.PoseLandmark.RIGHT_HIP),
        "left_knee": point(mp_pose.PoseLandmark.LEFT_KNEE),
        "right_knee": point(mp_pose.PoseLandmark.RIGHT_KNEE),
        "left_ankle": point(mp_pose.PoseLandmark.LEFT_ANKLE),
        "right_ankle": point(mp_pose.PoseLandmark.RIGHT_ANKLE),
        "left_foot": point(mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        "right_foot": point(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
    }
    points["left_hand"] = points["left_wrist"]
    points["right_hand"] = points["right_wrist"]
    return points


def video_to_wide_pattern(video_path, exercise_code, target_steps=80):
    if mp is None:
        raise RuntimeError("mediapipe belum ter-install.")

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Video tidak dapat dibuka.")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    frame_features = []
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if not result.pose_landmarks:
                continue

            points = mediapipe_result_to_points(result, mp_pose)
            aligned = body_aligned_normalization(points)
            if aligned is None:
                continue

            frame_features.append(extract_frame_features(aligned))

    cap.release()

    if len(frame_features) < 8:
        raise RuntimeError("Landmark tubuh terlalu sedikit terbaca. Pastikan seluruh tubuh terlihat jelas.")

    return frame_features_to_wide_pattern(frame_features, exercise_code, target_steps=target_steps)


def make_feedback(status, exercise_code, deviations, max_items=3):
    if status == "Correct":
        return ["Gerakan terdeteksi sesuai dengan pola Correct."]

    mapper = FEEDBACK_MAP.get(exercise_code, {})
    feedback = []
    for item in deviations[:max_items]:
        text = mapper.get(item["feature"])
        if text and text not in feedback:
            feedback.append(text)

    if not feedback:
        feedback.append("Gerakan terdeteksi berbeda dari pola Correct.")
    return feedback
