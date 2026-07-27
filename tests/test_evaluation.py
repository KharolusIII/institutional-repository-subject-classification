import numpy as np

from ir_subject_classification.evaluation import per_label_evaluation


def test_per_label_evaluation_includes_confusion_counts():
    y_true = np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]])
    y_pred = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]])
    scores = y_pred.astype(float)
    report = per_label_evaluation(y_true, y_pred, scores, ["A", "B"])
    first = report.set_index("label").loc["A"]
    assert first["true_negative"] == 1
    assert first["false_positive"] == 1
    assert first["false_negative"] == 1
    assert first["true_positive"] == 1
    assert first["predicted_support"] == 2
