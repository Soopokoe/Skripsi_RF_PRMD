import argparse
from pathlib import Path

import pandas as pd

from pose_pattern_utils import (
    parse_selected_exercises, read_position_file, row_to_joints,
    reconstruct_kinect_positions, paired_angle_path, build_augmentation_configs, apply_augmentation,
    infer_label, infer_exercise, infer_subject, infer_repetition, safe_stem,
)


def process_and_write(source_path, output_path, config):
    df = read_position_file(source_path)
    angles_df = read_position_file(paired_angle_path(source_path))
    rows = []
    raw_columns = df.shape[1]
    raw_joint_count = raw_columns // 3

    for frame_index in range(min(len(df), len(angles_df))):
        joints = row_to_joints(df.iloc[frame_index])
        angles = row_to_joints(angles_df.iloc[frame_index])
        if joints is None or angles is None:
            continue

        joints = reconstruct_kinect_positions(joints, angles)
        joints = apply_augmentation(joints, config)
        rows.append(joints.reshape(-1))

    if not rows:
        return 0, raw_columns, raw_joint_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, header=False)
    return len(rows), raw_columns, raw_joint_count


def main():
    parser = argparse.ArgumentParser(description="00 - Augmentasi dataset UI-PRMD untuk Random Forest dan XGBoost.")
    parser.add_argument("--dataset", required=True, help="Folder dataset UI-PRMD asli.")
    parser.add_argument("--output-dir", default="dataset_augmented_rf_xgb")
    parser.add_argument("--index-output", default="dataset_augmented_rf_xgb/augmented_index.csv")
    parser.add_argument("--selected-exercises", default="m01,m04,m05,m07")
    parser.add_argument("--max-files", type=int, default=None)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)
    index_output = Path(args.index_output)
    selected = parse_selected_exercises(args.selected_exercises)

    files = []
    for ext in ("*.txt", "*.csv"):
        files.extend(dataset_dir.rglob(ext))
    files = sorted(files)

    rows = []
    skipped = {
        "not_kinect_positions": 0,
        "no_label": 0,
        "no_exercise": 0,
        "not_selected": 0,
        "error": 0,
    }

    processed_original_files = 0

    for source_path in files:
        text = str(source_path).lower()

        if "kinect" not in text or "positions" not in text or "angles" in text:
            skipped["not_kinect_positions"] += 1
            continue

        label, label_name = infer_label(source_path)
        if label_name is None:
            skipped["no_label"] += 1
            continue

        exercise_code, exercise_name = infer_exercise(source_path)
        if exercise_code is None:
            skipped["no_exercise"] += 1
            continue

        if exercise_code not in selected:
            skipped["not_selected"] += 1
            continue

        configs = build_augmentation_configs(exercise_code)

        for config in configs:
            file_name = f"{safe_stem(source_path)}__{config['augmentation']}.csv"
            augmented_path = output_dir / exercise_code / label_name / file_name

            try:
                frame_count, raw_columns, raw_joint_count = process_and_write(source_path, augmented_path, config)

                if frame_count <= 0:
                    skipped["error"] += 1
                    continue

                rows.append({
                    "sequence_id": len(rows),
                    "source_file": str(source_path),
                    "augmented_file": str(augmented_path),
                    "exercise_code": exercise_code,
                    "exercise_name": exercise_name,
                    "subject": infer_subject(source_path),
                    "repetition": infer_repetition(source_path),
                    "label": label,
                    "label_name": label_name,
                    "augmentation": config["augmentation"],
                    "base_augmentation": config["base_augmentation"],
                    "camera_rotation_degree": config["camera_rotation_degree"],
                    "frame_count": frame_count,
                    "raw_columns": raw_columns,
                    "raw_joint_count": raw_joint_count,
                    "position_source": "kinect_reconstructed_world_positions",
                })

            except Exception as error:
                skipped["error"] += 1
                print(f"[SKIP ERROR] {source_path} | {config['augmentation']} | {error}")

        processed_original_files += 1
        if processed_original_files % 50 == 0:
            print(f"[INFO] file asli diproses: {processed_original_files} | file augmentasi: {len(rows)}")

        if args.max_files is not None and processed_original_files >= args.max_files:
            break

    if not rows:
        raise RuntimeError("Tidak ada file augmentasi yang berhasil dibuat.")

    index_output.parent.mkdir(parents=True, exist_ok=True)
    index_df = pd.DataFrame(rows)
    index_df.to_csv(index_output, index=False)

    summary_output = index_output.with_name(index_output.stem + "_summary.csv")
    summary = (
        index_df.groupby(["exercise_code", "exercise_name", "label_name", "base_augmentation"], dropna=False)
        .size()
        .reset_index(name="file_count")
    )
    summary.to_csv(summary_output, index=False)

    print("\n=== 00 AUGMENTASI SELESAI ===")
    print("Output folder:", output_dir)
    print("Index:", index_output)
    print("Summary:", summary_output)
    print("Jumlah file augmentasi:", len(index_df))
    print("\nDistribusi label:")
    print(index_df["label_name"].value_counts())
    print("\nDistribusi gerakan:")
    print(index_df["exercise_name"].value_counts())
    print("\nFile dilewati:")
    for key, value in skipped.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
