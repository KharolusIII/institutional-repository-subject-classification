"""Leakage-safe iterative multi-label train/validation/test splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def multilabel_split_coverage(
    frame: pd.DataFrame, calibration_required: bool = False, seed: int = 42
) -> pd.DataFrame:
    """Summarize label support for an existing, already-frozen split."""
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(frame["labels"])
    required = ["train", "validation", "test"] + (
        ["calibration"] if calibration_required else []
    )
    rows = []
    for position, label in enumerate(mlb.classes_):
        supports = {
            split: int(y[frame["split"].eq(split), position].sum())
            for split in ("train", "calibration", "validation", "test")
        }
        rows.append(
            {
                "label": label,
                "support_total": int(y[:, position].sum()),
                **{f"support_{split}": value for split, value in supports.items()},
                "present_in_all_splits": all(supports[split] > 0 for split in required),
                "split_seed": seed,
            }
        )
    return pd.DataFrame(rows)


def multilabel_train_validation_test_split(
    frame: pd.DataFrame,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    calibration_size: float = 0.0,
    seed: int = 42,
    max_tries: int = 40,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    if group_column and group_column in frame and frame[group_column].duplicated().any():
        grouped = (
            frame.groupby(group_column, as_index=False)["labels"]
            .agg(lambda rows: sorted({label for values in rows for label in values}))
        )
        grouped_split, _ = multilabel_train_validation_test_split(
            grouped,
            validation_size=validation_size,
            test_size=test_size,
            calibration_size=calibration_size,
            seed=seed,
            max_tries=max_tries,
        )
        assignment = grouped_split.set_index(group_column)["split"]
        result = frame.copy()
        result["split"] = result[group_column].map(assignment)
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(result["labels"])
        coverage = []
        for position, label in enumerate(mlb.classes_):
            supports = {
                split: int(y[result["split"].eq(split), position].sum())
                for split in ("train", "calibration", "validation", "test")
            }
            required = ["train", "validation", "test"] + (["calibration"] if calibration_size else [])
            coverage.append(
                {
                    "label": label,
                    "support_total": int(y[:, position].sum()),
                    **{f"support_{key}": value for key, value in supports.items()},
                    "present_in_all_splits": all(supports[key] > 0 for key in required),
                    "split_seed": int(grouped_split["split"].notna().any() and seed),
                }
            )
        return result, pd.DataFrame(coverage)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(frame["labels"])
    indices = np.arange(len(frame))
    best: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None = None
    best_missing = y.shape[1] + 1
    for attempt in range(max_tries):
        random_state = seed + attempt
        held_out_size = validation_size + test_size + calibration_size
        first = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=held_out_size, random_state=random_state
        )
        train_idx, temp_idx = next(first.split(indices, y))
        if calibration_size:
            calibration_rel, remainder_rel = next(
                MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=(validation_size + test_size) / held_out_size,
                    random_state=random_state,
                ).split(temp_idx, y[temp_idx])
            )
            calibration_idx = temp_idx[calibration_rel]
            remainder_idx = temp_idx[remainder_rel]
        else:
            calibration_idx = np.asarray([], dtype=int)
            remainder_idx = temp_idx
        second = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size / (validation_size + test_size), random_state=random_state
        )
        val_rel, test_rel = next(second.split(remainder_idx, y[remainder_idx]))
        val_idx, test_idx = remainder_idx[val_rel], remainder_idx[test_rel]
        missing = int(
            np.sum(
                (y[train_idx].sum(axis=0) == 0)
                | (y[val_idx].sum(axis=0) == 0)
                | (y[test_idx].sum(axis=0) == 0)
                | ((y[calibration_idx].sum(axis=0) == 0) if calibration_size else False)
            )
        )
        if missing < best_missing:
            best = train_idx, calibration_idx, val_idx, test_idx, random_state
            best_missing = missing
        if missing == 0:
            break
    if best is None:
        raise ValueError("Unable to split an empty dataset")
    train_idx, calibration_idx, val_idx, test_idx, used_seed = best
    result = frame.copy()
    result["split"] = "train"
    result.loc[result.index[val_idx], "split"] = "validation"
    result.loc[result.index[test_idx], "split"] = "test"
    if calibration_size:
        result.loc[result.index[calibration_idx], "split"] = "calibration"
    coverage = []
    for pos, label in enumerate(mlb.classes_):
        coverage.append(
            {
                "label": label,
                "support_total": int(y[:, pos].sum()),
                "support_train": int(y[train_idx, pos].sum()),
                "support_validation": int(y[val_idx, pos].sum()),
                "support_calibration": int(y[calibration_idx, pos].sum()),
                "support_test": int(y[test_idx, pos].sum()),
                "present_in_all_splits": bool(
                    y[train_idx, pos].sum()
                    and y[val_idx, pos].sum()
                    and y[test_idx, pos].sum()
                    and (y[calibration_idx, pos].sum() if calibration_size else True)
                ),
                "split_seed": used_seed,
            }
        )
    return result, pd.DataFrame(coverage)

