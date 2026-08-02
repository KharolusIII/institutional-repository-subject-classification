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


def test_exact_duplicate_content_groups_never_cross_splits():
    frame = pd.DataFrame(
        {
            "handle": [str(i) for i in range(80)],
            "content_group": [f"group-{i // 2}" if i < 20 else f"unique-{i}" for i in range(80)],
            "labels": [["a"] if i % 2 else ["b"] for i in range(80)],
        }
    )
    split, _ = multilabel_train_validation_test_split(
        frame, validation_size=0.15, test_size=0.15, calibration_size=0.10,
        group_column="content_group", seed=42,
    )
    assert split.groupby("content_group")["split"].nunique().max() == 1

