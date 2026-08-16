from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
FEATURE_DATA_PATH = MODULE_DIR / "artifacts_uiprmd_fk/kinect_domain_randomized_features.csv"

# Fitur-fitur ini tidak setara antara protokol UI-PRMD (melangkah ke samping)
# dan video implementasi (kaki sudah terbuka, lalu menekuk). Median dihitung
# dari seluruh data Kinect tanpa memakai label Correct/Incorrect.
UNRELIABLE_VIDEO_FEATURES = {
    ("m04", "standard"): {
        "foot_distance",
        "ankle_lateral_distance",
        "knee_lateral_distance",
        "trunk_angle",
    },
}


@lru_cache(maxsize=1)
def _reference_data():
    if not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(f"Data referensi adapter tidak ditemukan: {FEATURE_DATA_PATH}")
    return pd.read_csv(FEATURE_DATA_PATH)


def _matrix_for(exercise, mode, columns):
    data = _reference_data()
    group = data[(data["exercise_code"] == exercise) & (data["view_mode"] == mode)]
    if group.empty:
        raise ValueError(f"Referensi adapter kosong untuk {exercise}/{mode}.")
    return group.reindex(columns=columns).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)


def _base_mask(columns, bases):
    return np.asarray([
        any(column.startswith(base + "_t") or column.startswith(base + "__") for base in bases)
        for column in columns
    ], dtype=bool)


def _complete_sit_to_stand_protocol(values, columns):
    """Complete seated-to-standing input into Kinect's stand-sit-stand context."""
    output = values.copy()
    bases = sorted({re.split(r"_t\d{3}|__", column)[0] for column in columns})
    for base in bases:
        temporal_indexes = [
            index for index, column in enumerate(columns)
            if re.search(r"_t\d{3}$", column) and column.startswith(base + "_t")
        ]
        if not temporal_indexes:
            continue
        curve = output[temporal_indexes]
        x = np.linspace(0.0, 1.0, len(curve))
        half = len(curve) // 2
        completed = np.concatenate([
            np.interp(np.linspace(1.0, 0.0, half, endpoint=False), x, curve),
            np.interp(np.linspace(0.0, 1.0, len(curve) - half), x, curve),
        ])
        output[temporal_indexes] = completed

        column_index = {column: index for index, column in enumerate(columns)}
        start_key, end_key = f"{base}__raw_start", f"{base}__raw_end"
        if start_key in column_index and end_key in column_index:
            output[column_index[start_key]] = output[column_index[end_key]]
        peak_key, valley_key = f"{base}__peak_time", f"{base}__valley_time"
        if peak_key in column_index:
            output[column_index[peak_key]] = float(np.argmax(completed) / max(len(completed) - 1, 1))
        if valley_key in column_index:
            output[column_index[valley_key]] = float(np.argmin(completed) / max(len(completed) - 1, 1))
    return output


def _project_to_unlabeled_kinect(values, reference, columns, profile, blend, neighbor_count=15):
    if profile == "summary":
        feature_mask = np.asarray(["__" in column for column in columns])
    else:
        feature_mask = np.asarray([bool(re.search(r"_t\d{3}$", column)) for column in columns])
    q25 = np.percentile(reference[:, feature_mask], 25, axis=0)
    q75 = np.percentile(reference[:, feature_mask], 75, axis=0)
    scale = np.maximum(q75 - q25, 0.10)
    distance = np.mean(
        np.abs((reference[:, feature_mask] - values[feature_mask]) / scale), axis=1
    )
    count = min(neighbor_count, len(reference))
    nearest = np.argsort(distance)[:count]
    neighbor_median = np.median(reference[nearest], axis=0)
    projected = (1.0 - blend) * values + blend * neighbor_median
    return projected, count, round(float(np.mean(distance[nearest])), 4)


def adapt_model_input(row, exercise, mode, columns):
    """Map a MediaPipe row to the Kinect feature domain without labels or threshold edits."""
    values = np.asarray([row.get(column, np.nan) for column in columns], dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    reference = _matrix_for(exercise, mode, columns)
    details = {"method": "none", "masked_features": [], "neighbors": 0, "blend": 0.0}

    unreliable = UNRELIABLE_VIDEO_FEATURES.get((exercise, mode), set())
    if unreliable:
        mask = _base_mask(columns, unreliable)
        values[mask] = np.median(reference[:, mask], axis=0)
        details.update({
            "method": "reliability_mask",
            "masked_features": sorted(unreliable),
        })

    if exercise == "m04" and mode == "standard":
        blend = 0.25
        values, neighbor_count, mean_distance = _project_to_unlabeled_kinect(
            values, reference, columns, "summary", blend
        )
        details.update({
            "method": "reliability_mask+unlabeled_kinect_summary_manifold",
            "neighbors": int(neighbor_count),
            "blend": blend,
            "mean_neighbor_distance": mean_distance,
        })

    # Deep Squat video memiliki bentuk kurva yang benar tetapi amplitudo sensor
    # berada di luar manifold Kinect. Cari tetangga dari semua data (tanpa label).
    if exercise == "m01" and mode == "standard":
        blend = 0.50
        values, neighbor_count, mean_distance = _project_to_unlabeled_kinect(
            values, reference, columns, "temporal", blend
        )
        details.update({
            "method": "unlabeled_kinect_manifold",
            "neighbors": int(neighbor_count),
            "blend": blend,
            "mean_neighbor_distance": mean_distance,
        })

    if exercise == "m05" and mode == "one_side":
        values = _complete_sit_to_stand_protocol(values, columns)
        blend = 0.25
        values, neighbor_count, mean_distance = _project_to_unlabeled_kinect(
            values, reference, columns, "temporal", blend
        )
        details.update({
            "method": "protocol_completion+unlabeled_kinect_manifold",
            "neighbors": int(neighbor_count),
            "blend": blend,
            "mean_neighbor_distance": mean_distance,
        })

    if exercise == "m07" and mode == "standard":
        blend = 0.25
        values, neighbor_count, mean_distance = _project_to_unlabeled_kinect(
            values, reference, columns, "summary", blend
        )
        details.update({
            "method": "unlabeled_kinect_summary_manifold",
            "neighbors": int(neighbor_count),
            "blend": blend,
            "mean_neighbor_distance": mean_distance,
        })

    return {column: float(value) for column, value in zip(columns, values)}, details
