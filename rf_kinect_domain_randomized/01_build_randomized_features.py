from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from domain_randomization import VARIANTS, extract_randomized_features  # noqa: E402
from pose_pattern_utils import (  # noqa: E402
    extract_named_points_from_kinect_joints,
    frame_features_to_wide_pattern,
    paired_angle_path,
    read_position_file,
    reconstruct_kinect_positions,
    row_to_joints,
)


def main():
    parser = argparse.ArgumentParser(description="Bangun fitur Kinect dengan domain randomization kamera.")
    parser.add_argument("--index", default=str(PROJECT_DIR / "dataset_augmented_rf_xgb_fixed/augmented_index.csv"))
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "artifacts_uiprmd_fk/kinect_domain_randomized_features.csv"),
    )
    parser.add_argument("--target-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--repetitions-per-subject-label", type=int, default=5,
        help="Sampling seimbang; 5 memakai 400 rekaman aktif sambil mempertahankan semua subjek.",
    )
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    index = index[
        (index["base_augmentation"] == "original")
        & (pd.to_numeric(index["camera_rotation_degree"], errors="coerce") == 0)
    ].drop_duplicates("source_file")
    if args.repetitions_per_subject_label:
        index = (
            index.groupby(["exercise_code", "label", "subject"], group_keys=False)
            .sample(n=args.repetitions_per_subject_label, random_state=args.seed)
            .sort_values(["exercise_code", "label", "subject", "repetition"])
        )
    if args.max_files:
        index = index.head(args.max_files)

    rows = []
    skipped = 0
    for file_number, (_, item) in enumerate(index.iterrows(), start=1):
        # Baca sumber UI-PRMD asli dan rekonstruksi rantai parent dengan urutan
        # 22 joint resmi. augmented_file lama tidak dipakai karena pernah dibuat
        # dengan pemetaan joint generik Kinect SDK yang tidak sesuai UI-PRMD.
        source_path = Path(item["source_file"])
        raw = read_position_file(source_path)
        raw_angles = read_position_file(paired_angle_path(source_path))
        point_frames = []
        frame_count = min(len(raw), len(raw_angles))
        for frame_index in range(frame_count):
            raw_row = raw.iloc[frame_index]
            joints = row_to_joints(raw_row)
            angles = row_to_joints(raw_angles.iloc[frame_index])
            if joints is None or angles is None:
                continue
            joints = reconstruct_kinect_positions(joints, angles)
            points = extract_named_points_from_kinect_joints(joints)
            if points is not None:
                point_frames.append(points)
        if len(point_frames) < 8:
            skipped += 1
            continue

        for variant_number, config in enumerate(VARIANTS):
            if config.get("exercises") and item["exercise_code"] not in config["exercises"]:
                continue
            rng = np.random.default_rng(args.seed + int(item["sequence_id"]) * 17 + variant_number)
            randomized = extract_randomized_features(
                point_frames, item["exercise_code"], config, rng
            )
            if len(randomized) < 8:
                continue
            feature_row, names, start, end, used = frame_features_to_wide_pattern(
                randomized, item["exercise_code"], args.target_steps
            )
            output_row = {
                "source_file": item["source_file"],
                "exercise_code": item["exercise_code"],
                "exercise_name": item["exercise_name"],
                "subject": item["subject"],
                "repetition": item["repetition"],
                "label": int(item["label"]),
                "label_name": item["label_name"],
                "view_mode": config["mode"],
                "domain_variant": config["name"],
                "feature_names": ",".join(names),
                "used_frame_count": used,
                "segment_start": start,
                "segment_end": end,
                "target_steps": args.target_steps,
            }
            output_row.update(feature_row)
            rows.append(output_row)
        if file_number % 50 == 0:
            print(f"files={file_number}/{len(index)} | feature rows={len(rows)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    summary = frame.groupby(["exercise_code", "view_mode", "label_name"]).size().reset_index(name="rows")
    summary.to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    print(f"\nOutput: {output}")
    print(f"Rows: {len(frame)} | columns: {len(frame.columns)} | skipped: {skipped}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
