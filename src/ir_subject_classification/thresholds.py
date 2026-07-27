"""Validation-only threshold optimization."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def apply_thresholds(scores: np.ndarray, thresholds: float | np.ndarray) -> np.ndarray:
    return (np.asarray(scores) >= thresholds).astype(int)


def optimize_global_threshold(
    y_true: np.ndarray, scores: np.ndarray, candidates: np.ndarray | None = None
) -> tuple[float, float]:
    candidates = candidates if candidates is not None else np.linspace(0.05, 0.95, 19)
    evaluated = [(float(t), f1_score(y_true, apply_thresholds(scores, t), average="macro", zero_division=0)) for t in candidates]
    return max(evaluated, key=lambda item: (item[1], -abs(item[0] - 0.5)))


def optimize_per_label_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    global_threshold: float,
    minimum_support: int = 20,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    candidates = candidates if candidates is not None else np.linspace(0.05, 0.95, 19)
    result = np.full(y_true.shape[1], global_threshold, dtype=float)
    for label in range(y_true.shape[1]):
        if y_true[:, label].sum() < minimum_support:
            continue
        values = [
            f1_score(y_true[:, label], scores[:, label] >= threshold, zero_division=0) for threshold in candidates
        ]
        result[label] = float(candidates[int(np.argmax(values))])
    return result

