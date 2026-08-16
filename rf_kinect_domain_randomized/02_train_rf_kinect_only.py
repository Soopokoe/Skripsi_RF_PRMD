from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from rf_calibration import calibrated_probability, choose_balanced_threshold, logit


META = {
    "source_file", "exercise_code", "exercise_name", "subject", "repetition", "label",
    "label_name", "view_mode", "domain_variant", "feature_names", "used_frame_count",
    "segment_start", "segment_end", "target_steps",
}


def columns_for_mode(group, stride, profile="temporal_summary"):
    columns = []
    for column in group.columns:
        if column in META or not pd.api.types.is_numeric_dtype(group[column]):
            continue
        match = re.search(r"_t(\d{3})$", column)
        base = re.split(r"_t\d{3}|__", column)[0]
        if profile == "upper_body_cross_sensor":
            allowed = {
                "dominant_shoulder_angle", "dominant_elbow_angle", "dominant_hand_height",
                "trunk_angle", "shoulder_height_difference",
            }
            if base not in allowed:
                continue
            # MediaPipe world-depth memberi offset sudut torso yang berbeda dari
            # Kinect. Pertahankan bentuk temporal torso, bukan amplitudo absolut.
            if base == "trunk_angle" and not match:
                continue
        if profile == "summary_only" and match:
            continue
        if profile == "temporal_only" and not match:
            continue
        if match and int(match.group(1)) % stride != 0:
            continue
        if group[column].notna().any():
            columns.append(column)
    return columns


def build_rf(seed=42, trees=800):
    return RandomForestClassifier(
        n_estimators=trees, max_depth=14, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )


def main():
    module_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="RF Kinect-only dengan OOF probability calibration.")
    project_dir = module_dir.parent
    parser.add_argument(
        "--data", default=str(project_dir / "artifacts_uiprmd_fk/kinect_domain_randomized_features.csv")
    )
    parser.add_argument("--output-dir", default=str(project_dir / "artifacts_uiprmd_fk/models"))
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument(
        "--selected-exercises", default="",
        help="Kode gerakan dipisahkan koma; kosong berarti melatih seluruh gerakan.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_exercises = {item.strip() for item in args.selected_exercises.split(",") if item.strip()}
    index_path = output / "model_index.json"
    metrics_path = output / "training_metrics.json"
    if selected_exercises and index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    else:
        index = {"version": 2, "training_domain": "kinect_only_uiprmd_forward_kinematics", "exercises": {}}
        metrics = {}

    for (exercise, mode), group in data.groupby(["exercise_code", "view_mode"]):
        if selected_exercises and exercise not in selected_exercises:
            continue
        y = group["label"].astype(int).to_numpy()
        subjects = group["subject"].astype(str).to_numpy()
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        profiles = [("temporal_summary", args.temporal_stride)]
        if exercise == "m07":
            profiles = [("upper_body_cross_sensor", args.temporal_stride)]
        if exercise in {"m01", "m04"} and mode == "standard":
            profiles.extend([("summary_only", args.temporal_stride), ("temporal_summary_sparse", 8)])
        profile_results = []
        for profile, stride in profiles:
            column_profile = "temporal_summary" if profile == "temporal_summary_sparse" else profile
            candidate_columns = columns_for_mode(group, stride, column_profile)
            candidate_X = group[candidate_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            candidate_oof = np.zeros(len(group), dtype=float)
            for fold, (train_index, validation_index) in enumerate(splitter.split(candidate_X, y, subjects)):
                model = build_rf(42 + fold, trees=400)
                model.fit(candidate_X.iloc[train_index], y[train_index])
                candidate_oof[validation_index] = model.predict_proba(candidate_X.iloc[validation_index])[:, 1]
            candidate_calibrator = LogisticRegression(C=1.0, random_state=42)
            candidate_calibrator.fit(logit(candidate_oof), y)
            candidate_calibrated = calibrated_probability(candidate_calibrator, candidate_oof)
            candidate_threshold = choose_balanced_threshold(y, candidate_calibrated)
            candidate_prediction = (candidate_calibrated >= candidate_threshold).astype(int)
            candidate_ba = float(balanced_accuracy_score(y, candidate_prediction))
            profile_results.append({
                "profile": profile, "stride": stride, "columns": candidate_columns,
                "raw_oof": candidate_oof, "calibrator": candidate_calibrator,
                "calibrated": candidate_calibrated, "threshold": candidate_threshold,
                "prediction": candidate_prediction, "balanced_accuracy": candidate_ba,
            })
        selected = max(profile_results, key=lambda item: item["balanced_accuracy"])
        columns = selected["columns"]
        X = group[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_oof = selected["raw_oof"]
        calibrator = selected["calibrator"]
        calibrated = selected["calibrated"]
        threshold = selected["threshold"]
        prediction = selected["prediction"]
        mode_metrics = {
            "rows": len(group),
            "subjects": int(group["subject"].nunique()),
            "features": len(columns),
            "threshold": threshold,
            "raw_probability_median_correct": float(np.median(raw_oof[y == 1])),
            "raw_probability_median_incorrect": float(np.median(raw_oof[y == 0])),
            "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            "selected_profile": selected["profile"],
            "profile_search": {
                item["profile"]: {
                    "balanced_accuracy": item["balanced_accuracy"],
                    "threshold": item["threshold"],
                    "features": len(item["columns"]),
                } for item in profile_results
            },
        }

        final_model = build_rf(42)
        final_model.fit(X, y)
        model_path = output / f"rf_{exercise}_{mode}.joblib"
        calibrator_path = output / f"calibrator_{exercise}_{mode}.joblib"
        joblib.dump(final_model, model_path, compress=3)
        joblib.dump(calibrator, calibrator_path, compress=3)
        exercise_item = index["exercises"].setdefault(exercise, {"modes": {}})
        exercise_item["modes"][mode] = {
            "model_path": str(model_path.resolve()),
            "calibrator_path": str(calibrator_path.resolve()),
            "feature_columns": columns,
            "threshold": threshold,
            "feature_profile": selected["profile"],
        }
        metrics.setdefault(exercise, {})[mode] = mode_metrics
        print(
            f"{exercise}/{mode} | OOF BA={mode_metrics['balanced_accuracy']:.3f} | "
            f"threshold={threshold:.3f} | profile={selected['profile']} | features={len(columns)}"
        )

    index["version"] = 2
    index["training_domain"] = "kinect_only_uiprmd_forward_kinematics"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nIndex: {output / 'model_index.json'}")


if __name__ == "__main__":
    main()
