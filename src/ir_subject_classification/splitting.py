"""Leakage-safe iterative multi-label train/validation/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def multilabel_train_validation_test_split(
    frame: pd.DataFrame,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    max_tries: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(frame["labels"])
    indices = np.arange(len(frame))
    best: tuple[np.ndarray, np.ndarray, np.ndarray, int] | None = None
    best_missing = y.shape[1] + 1
    for attempt in range(max_tries):
        random_state = seed + attempt
        first = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=validation_size + test_size, random_state=random_state
        )
        train_idx, temp_idx = next(first.split(indices, y))
        second = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size / (validation_size + test_size), random_state=random_state
        )
        val_rel, test_rel = next(second.split(temp_idx, y[temp_idx]))
        val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]
        missing = int(
            np.sum(
                (y[train_idx].sum(axis=0) == 0)
                | (y[val_idx].sum(axis=0) == 0)
                | (y[test_idx].sum(axis=0) == 0)
            )
        )
        if missing < best_missing:
            best = train_idx, val_idx, test_idx, random_state
            best_missing = missing
        if missing == 0:
            break
    if best is None:
        raise ValueError("Unable to split an empty dataset")
    train_idx, val_idx, test_idx, used_seed = best
    result = frame.copy()
    result["split"] = "train"
    result.loc[result.index[val_idx], "split"] = "validation"
    result.loc[result.index[test_idx], "split"] = "test"
    coverage = []
    for pos, label in enumerate(mlb.classes_):
        coverage.append(
            {
                "label": label,
                "support_total": int(y[:, pos].sum()),
                "support_train": int(y[train_idx, pos].sum()),
                "support_validation": int(y[val_idx, pos].sum()),
                "support_test": int(y[test_idx, pos].sum()),
                "present_in_all_splits": bool(
                    y[train_idx, pos].sum() and y[val_idx, pos].sum() and y[test_idx, pos].sum()
                ),
                "split_seed": used_seed,
            }
        )
    return result, pd.DataFrame(coverage)

