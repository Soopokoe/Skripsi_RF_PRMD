from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
TEMPLATE_DIR = PROJECT_DIR / "template_matching_multi_angle"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(TEMPLATE_DIR))

from rf_calibration import calibrated_probability  # noqa: E402
from input_adapter import adapt_model_input  # noqa: E402
from video_extractor import video_to_visibility_aware_pattern  # noqa: E402


VIDEOS = {
    "m01": [
        PROJECT_DIR / "Video/Deep Squat 1.mp4",
        PROJECT_DIR / "Video/Deep Squat Rama.mp4",
    ],
    "m04": [
        PROJECT_DIR / "Video/Side Lunge 1.mp4",
        PROJECT_DIR / "Video/Side Lunge 1_flipped.mp4",
        PROJECT_DIR / "Video/Side Lunge 2.mp4",
        PROJECT_DIR / "Video/Side Lunge Rama (-).mp4",
        PROJECT_DIR / "Video/Side Lunge Rama2 (-).mp4",
    ],
    "m05": [
        PROJECT_DIR / "Video/Sit to Stand 1.mp4",
        PROJECT_DIR / "Video/Sit to Stand 1_flipped.mp4",
        PROJECT_DIR / "Video/Sit To Stand Rama.mp4",
    ],
    "m07": [
        PROJECT_DIR / "Video/Standing Shoulder Abduction (ABD AROM).mp4",
        PROJECT_DIR / "Video/Standing Shoulder ABD 2.mp4",
        PROJECT_DIR / "Video/Standing Shoulder ABD 3.mp4",
    ],
}


def resolve_artifact(path_value, model_index_path):
    configured_path = Path(path_value)
    candidates = (configured_path, Path(model_index_path).resolve().parent / configured_path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Artefak model tidak ditemukan: {configured_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Uji RF Kinect-only dengan adapter input MediaPipe.")
    parser.add_argument("--index", default=str(PROJECT_DIR / "artifacts_uiprmd_fk/models/model_index.json"))
    parser.add_argument("--output", default=str(PROJECT_DIR / "artifacts_uiprmd_fk/video_test_results.json"))
    parser.add_argument("--exercises", default="", help="Batasi kode gerakan, dipisahkan koma.")
    parser.add_argument(
        "--video-name-contains", default="",
        help="Uji hanya nama video yang memuat salah satu teks, dipisahkan koma.",
    )
    args = parser.parse_args()
    selected = {item.strip() for item in args.exercises.split(",") if item.strip()}
    name_filters = [item.strip().lower() for item in args.video_name_contains.split(",") if item.strip()]
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    results = []

    for exercise, paths in VIDEOS.items():
        if selected and exercise not in selected:
            continue
        for path in paths:
            if not path.exists():
                print(f"Lewati video yang tidak tersedia: {path}")
                continue
            if name_filters and not any(token in path.name.lower() for token in name_filters):
                continue
            _, _, _, _, _, diagnostics, candidates = video_to_visibility_aware_pattern(path, exercise, 80)
            mode = "one_side" if diagnostics["view_mode"] == "one_side" else "standard"
            info = index["exercises"][exercise]["modes"][mode]
            model = joblib.load(resolve_artifact(info["model_path"], args.index))
            calibrator = joblib.load(resolve_artifact(info["calibrator_path"], args.index))
            repetitions = []
            for number, candidate in enumerate(candidates, start=1):
                adapted_row, model_input_adapter = adapt_model_input(
                    candidate["row"], exercise, mode, info["feature_columns"]
                )
                frame = pd.DataFrame([
                    adapted_row
                ], columns=info["feature_columns"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                raw_probability = float(model.predict_proba(frame)[0, 1])
                probability = float(calibrated_probability(calibrator, [raw_probability])[0])
                repetitions.append({
                    "repetition": number,
                    "raw_probability": round(raw_probability, 6),
                    "calibrated_probability": round(probability, 6),
                    "status": "Correct" if probability >= info["threshold"] else "Incorrect",
                    "model_input_adapter": model_input_adapter,
                })
            median_probability = float(np.median([item["calibrated_probability"] for item in repetitions]))
            result = {
                "exercise": exercise,
                "video": str(path.resolve()),
                "mode": mode,
                "status": "Correct" if median_probability >= info["threshold"] else "Incorrect",
                "median_calibrated_probability": round(median_probability, 6),
                "kinect_only_threshold": round(float(info["threshold"]), 6),
                "repetitions": repetitions,
                "video_used_for_training_or_threshold": False,
                "input_adapter_version": diagnostics.get("input_adapter_version"),
                "input_adapter_features": diagnostics.get("input_adapter_features", []),
            }
            results.append(result)
            print(
                f"{path.name} | {mode} | {result['status']} | "
                f"Pcal={median_probability:.3f} | T={info['threshold']:.3f}"
            )

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nHasil: {args.output}")


if __name__ == "__main__":
    main()
