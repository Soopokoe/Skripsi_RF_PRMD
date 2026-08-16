from __future__ import annotations

import numpy as np


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(probability / (1.0 - probability)).reshape(-1, 1)


def calibrated_probability(calibrator, probability):
    return calibrator.predict_proba(logit(probability))[:, 1]


def choose_balanced_threshold(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(probabilities)
    if len(candidates) > 1:
        candidates = (candidates[:-1] + candidates[1:]) / 2.0
    best = None
    for threshold in candidates:
        prediction = (probabilities >= threshold).astype(int)
        correct_recall = np.mean(prediction[y_true == 1] == 1)
        incorrect_recall = np.mean(prediction[y_true == 0] == 0)
        balanced = (correct_recall + incorrect_recall) / 2.0
        candidate = (balanced, -abs(float(threshold) - 0.5))
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold))
    return best[1]
