"""Multilabel classification and ranking metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([y_true[row, top[row]].sum() / k for row in range(len(y_true))]))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    values = []
    for row in range(len(y_true)):
        positives = y_true[row].sum()
        values.append(y_true[row, top[row]].sum() / positives if positives else 0.0)
    return float(np.mean(values))


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, float]:
    result = {
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_samples": f1_score(y_true, y_pred, average="samples", zero_division=0),
        "jaccard_micro": jaccard_score(y_true, y_pred, average="micro", zero_division=0),
        "jaccard_macro": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
    }
    if scores is not None:
        try:
            result["average_precision_micro"] = average_precision_score(y_true, scores, average="micro")
            result["average_precision_macro"] = average_precision_score(y_true, scores, average="macro")
        except ValueError:
            result["average_precision_micro"] = float("nan")
            result["average_precision_macro"] = float("nan")
        for k in (1, 3, 5):
            result[f"precision_at_{k}"] = precision_at_k(y_true, scores, k)
            result[f"recall_at_{k}"] = recall_at_k(y_true, scores, k)
    return {key: float(value) for key, value in result.items()}


def bootstrap_confidence_intervals(
    y_true: np.ndarray, y_pred: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 42
) -> list[dict[str, float | str]]:
    generator = np.random.default_rng(seed)
    values = {"f1_macro": [], "f1_micro": []}
    for _ in range(n_resamples):
        index = generator.integers(0, len(y_true), len(y_true))
        values["f1_macro"].append(f1_score(y_true[index], y_pred[index], average="macro", zero_division=0))
        values["f1_micro"].append(f1_score(y_true[index], y_pred[index], average="micro", zero_division=0))
    alpha = (1 - confidence) / 2
    return [
        {
            "metric": metric,
            "estimate": float(f1_score(y_true, y_pred, average=metric.split("_")[1], zero_division=0)),
            "ci_lower": float(np.quantile(samples, alpha)),
            "ci_upper": float(np.quantile(samples, 1 - alpha)),
            "confidence": confidence,
            "n_resamples": n_resamples,
        }
        for metric, samples in values.items()
    ]

