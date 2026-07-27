import numpy as np

from ir_subject_classification.metrics import multilabel_metrics, precision_at_k, recall_at_k


def test_precision_and_recall_at_k():
    y = np.array([[1, 0, 1], [0, 1, 0]])
    scores = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.1]])
    assert precision_at_k(y, scores, 1) == 1.0
    assert recall_at_k(y, scores, 1) == 0.75
    result = multilabel_metrics(y, y, scores)
    assert result["subset_accuracy"] == 1.0
    assert "precision_at_3" in result

