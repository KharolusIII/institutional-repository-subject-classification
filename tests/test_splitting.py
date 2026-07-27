import pandas as pd

from ir_subject_classification.splitting import multilabel_train_validation_test_split


def test_multilabel_split_preserves_well_supported_labels():
    frame = pd.DataFrame(
        {
            "handle": [str(i) for i in range(60)],
            "labels": [["a", "b"] if i % 6 == 0 else ["a"] if i % 2 == 0 else ["b"] for i in range(60)],
        }
    )
    split, coverage = multilabel_train_validation_test_split(frame, seed=42)
    assert set(split["split"]) == {"train", "validation", "test"}
    assert coverage["present_in_all_splits"].all()
    assert sorted(split["handle"]) == sorted(frame["handle"])

