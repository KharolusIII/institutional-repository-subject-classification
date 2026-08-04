from ir_subject_classification.classifiers import (
    convergence_diagnostics,
    create_classifier,
)


def test_named_linear_svc_audit_variant_and_diagnostics():
    classifier = create_classifier(
        "linear_svc_c05_tol5e4",
        {"C": 0.5, "tol": 0.0005, "max_iter": 1000, "dual": "auto"},
    )
    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]
    y = [[1, 0], [0, 1], [1, 1], [0, 0]]
    classifier.fit(x, y)

    rows = convergence_diagnostics(classifier, ["A", "B"])

    assert [row["label"] for row in rows] == ["A", "B"]
    assert all(row["iterations"] is not None for row in rows)
    assert all(row["max_iter"] == 1000 for row in rows)
    assert all(row["converged"] is True for row in rows)
