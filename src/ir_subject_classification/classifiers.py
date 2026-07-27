"""One-vs-Rest classifier factories."""

from __future__ import annotations

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


def create_classifier(name: str, params: dict[str, object] | None = None, random_state: int = 42):
    params = dict(params or {})
    if name == "logreg":
        params.setdefault("solver", "liblinear")
        params.setdefault("max_iter", 3000)
        params.setdefault("random_state", random_state)
        base = LogisticRegression(**params)
    elif name == "linear_svc":
        params.setdefault("max_iter", 200000)
        params.setdefault("random_state", random_state)
        base = LinearSVC(**params)
    elif name == "sgd":
        params.setdefault("loss", "log_loss")
        params.setdefault("max_iter", 3000)
        params.setdefault("tol", 1e-3)
        params.setdefault("random_state", random_state)
        base = SGDClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {name}")
    return OneVsRestClassifier(clone(base))


def prediction_scores(classifier, x):
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(x), "probabilities"
    if hasattr(classifier, "decision_function"):
        return classifier.decision_function(x), "decision_scores"
    raise TypeError("Classifier exposes neither predict_proba nor decision_function")

