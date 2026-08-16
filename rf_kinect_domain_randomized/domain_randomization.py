from __future__ import annotations

import numpy as np

from pose_pattern_utils import angle_3d, body_aligned_normalization, extract_frame_features


TWO_SIDE_FEATURES = {
    "hand_height_difference", "shoulder_height_difference", "hip_height_difference",
    "knee_height_difference", "ankle_height_difference", "foot_distance", "hand_distance",
    "ankle_lateral_distance", "ankle_sagittal_distance", "knee_lateral_distance",
    "knee_sagittal_distance",
}


VARIANTS = (
    {"name": "front_sensor", "mode": "standard", "yaw": 8, "pitch": 4, "noise": 0.010, "depth": 0.55, "warp": 0.85},
    {"name": "oblique_left", "mode": "standard", "yaw": 35, "pitch": -4, "noise": 0.012, "depth": 0.45, "warp": 1.15},
    {"name": "oblique_right", "mode": "standard", "yaw": -35, "pitch": 4, "noise": 0.012, "depth": 0.45, "warp": 0.92},
    {"name": "side_left", "mode": "one_side", "yaw": 80, "pitch": 0, "noise": 0.015, "depth": 0.20, "warp": 0.85, "visible": "left"},
    {"name": "side_right", "mode": "one_side", "yaw": -80, "pitch": 0, "noise": 0.015, "depth": 0.20, "warp": 1.15, "visible": "right"},
    {"name": "front_long_hold", "mode": "standard", "yaw": 0, "pitch": 2, "noise": 0.010, "depth": 0.55, "warp": 1.0, "hold_fraction": 0.50, "exercises": {"m01", "m04"}},
    {"name": "oblique_long_hold", "mode": "standard", "yaw": 35, "pitch": -3, "noise": 0.014, "depth": 0.40, "warp": 1.0, "hold_fraction": 0.80, "exercises": {"m01", "m04"}},
)


def _rotation(yaw, pitch):
    y = np.deg2rad(yaw)
    x = np.deg2rad(pitch)
    rotate_y = np.array([
        [np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]
    ])
    rotate_x = np.array([
        [1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]
    ])
    return rotate_x @ rotate_y


def simulate_camera(points, config, rng):
    left_hip, right_hip = points["left_hip"], points["right_hip"]
    left_shoulder, right_shoulder = points["left_shoulder"], points["right_shoulder"]
    center = (left_hip + right_hip) / 2.0
    torso = np.linalg.norm((left_shoulder + right_shoulder) / 2.0 - center)
    torso = max(float(torso), 1e-6)
    rotation = _rotation(config["yaw"], config["pitch"])
    focal = torso * 5.0
    output = {}
    for name, point in points.items():
        rotated = rotation @ (np.asarray(point, dtype=float) - center)
        perspective = focal / max(focal + rotated[2], focal * 0.35)
        rotated[0] *= perspective
        rotated[1] *= perspective
        rotated[2] *= config["depth"]
        distal = 1.5 if any(token in name for token in ("wrist", "hand", "ankle", "foot")) else 1.0
        rotated += rng.normal(0.0, config["noise"] * torso * distal, size=3)
        output[name] = rotated + center
    return output


def visible_side_features(aligned, visible):
    features = extract_frame_features(aligned)
    hidden = "right" if visible == "left" else "left"
    features[f"{hidden}_knee_angle"] = features[f"{visible}_knee_angle"]
    features[f"{hidden}_hip_angle"] = features[f"{visible}_hip_angle"]
    features[f"{hidden}_shoulder_angle"] = features[f"{visible}_shoulder_angle"]
    features[f"{hidden}_elbow_angle"] = features[f"{visible}_elbow_angle"]
    features["hip_to_ankle_height"] = aligned[f"{visible}_hip"][1] - aligned[f"{visible}_ankle"][1]
    features["knee_mean_height"] = aligned[f"{visible}_knee"][1]
    hand = aligned.get(f"{visible}_hand", aligned[f"{visible}_wrist"])
    features["hand_mean_height"] = hand[1]
    shoulder, hip = aligned[f"{visible}_shoulder"], aligned[f"{visible}_hip"]
    features["trunk_angle"] = 180.0 - angle_3d(
        shoulder + np.array([0.0, 1.0, 0.0]), shoulder, hip
    )
    for name in TWO_SIDE_FEATURES:
        features[name] = np.nan
    return features


def temporal_warp(frame_features, exponent, hold_fraction=0.0):
    count = len(frame_features)
    if count < 8:
        return frame_features
    positions = np.linspace(0.0, 1.0, count) ** exponent
    indexes = np.clip(np.round(positions * (count - 1)).astype(int), 0, count - 1)
    warped = [frame_features[index] for index in indexes]
    if hold_fraction > 0:
        knee = np.asarray([
            (item["left_knee_angle"] + item["right_knee_angle"]) / 2.0 for item in warped
        ])
        center = int(np.nanargmin(knee))
        hold = [warped[center]] * max(3, int(round(count * hold_fraction)))
        warped = warped[:center] + hold + warped[center:]
    return warped


def extract_randomized_features(point_frames, exercise_code, config, rng):
    frame_features = []
    for points in point_frames:
        simulated = simulate_camera(points, config, rng)
        aligned = body_aligned_normalization(simulated)
        if aligned is None:
            continue
        if config["mode"] == "one_side":
            features = visible_side_features(aligned, config["visible"])
        else:
            features = extract_frame_features(aligned)
        frame_features.append(features)
    default_hold = 0.10 if exercise_code in {"m01", "m04"} else 0.0
    return temporal_warp(frame_features, config["warp"], config.get("hold_fraction", default_hold))
