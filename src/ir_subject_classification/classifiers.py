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
    elif name == "linear_svc" or name.startswith("linear_svc_"):
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


def convergence_diagnostics(classifier, labels: list[str]) -> list[dict[str, object]]:
    """Return one convergence record for every one-vs-rest estimator.

    Scikit-learn exposes ``n_iter_`` on each fitted linear estimator.  Recording
    it per label makes convergence warnings auditable instead of leaving a
    single ambiguous warning in a long Colab log.
    """
    rows: list[dict[str, object]] = []
    estimators = getattr(classifier, "estimators_", [])
    for index, estimator in enumerate(estimators):
        raw_iterations = getattr(estimator, "n_iter_", None)
        if raw_iterations is None:
            iterations = None
        else:
            values = getattr(raw_iterations, "ravel", lambda: [raw_iterations])()
            iterations = int(max(values))
        max_iter = getattr(estimator, "max_iter", None)
        converged = None if iterations is None or max_iter is None else iterations < int(max_iter)
        rows.append(
            {
                "label": labels[index] if index < len(labels) else str(index),
                "iterations": iterations,
                "max_iter": int(max_iter) if max_iter is not None else None,
                "converged": converged,
            }
        )
    return rows


def prediction_scores(classifier, x):
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(x), "probabilities"
    if hasattr(classifier, "decision_function"):
        return classifier.decision_function(x), "decision_scores"
    raise TypeError("Classifier exposes neither predict_proba nor decision_function")

