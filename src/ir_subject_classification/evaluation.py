"""Evaluation tables by label and language."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from .metrics import multilabel_metrics


def per_label_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    support_train: np.ndarray | None = None,
    support_validation: np.ndarray | None = None,
    thresholds: float | np.ndarray = 0.5,
) -> pd.DataFrame:
    precision, recall, f1, support_test = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    threshold_values = np.full(len(labels), thresholds) if np.isscalar(thresholds) else np.asarray(thresholds)
    rows = []
    for index, label in enumerate(labels):
        try:
            ap = average_precision_score(y_true[:, index], scores[:, index])
        except ValueError:
            ap = float("nan")
        rows.append(
            {
                "label": label,
                "support_train": int(support_train[index]) if support_train is not None else None,
                "support_validation": int(support_validation[index]) if support_validation is not None else None,
                "support_test": int(support_test[index]),
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "average_precision": ap,
                "threshold": threshold_values[index],
            }
        )
    return pd.DataFrame(rows)


def support_f1_correlations(per_label: pd.DataFrame) -> pd.DataFrame:
    usable = per_label.dropna(subset=["support_test", "f1"])
    if len(usable) < 2:
        return pd.DataFrame(columns=["correlation", "value", "p_value"])
    pearson = pearsonr(np.log1p(usable["support_test"]), usable["f1"])
    spearman = spearmanr(usable["support_test"], usable["f1"])
    return pd.DataFrame(
        [
            {"correlation": "pearson_log1p_support_f1", "value": pearson.statistic, "p_value": pearson.pvalue},
            {"correlation": "spearman_support_f1", "value": spearman.statistic, "p_value": spearman.pvalue},
        ]
    )


def language_performance(
    languages: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    minimum_support: int = 30,
) -> pd.DataFrame:
    rows = []
    for language, indices in languages.groupby(languages).groups.items():
        index = np.asarray(list(indices))
        if len(index) < minimum_support:
            continue
        metrics = multilabel_metrics(y_true[index], y_pred[index], scores[index])
        rows.append(
            {
                "language": language,
                "N": len(index),
                **{key: metrics[key] for key in ("f1_micro", "f1_macro", "precision_at_3", "recall_at_3", "precision_at_5", "recall_at_5")},
            }
        )
    return pd.DataFrame(rows)

