"""Multi-label stratified sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def multilabel_sample(frame: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    if n is None or n >= len(frame):
        return frame.copy().reset_index(drop=True)
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(frame["labels"])
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - n / len(frame), random_state=seed)
    keep, _ = next(splitter.split(np.arange(len(frame)), y))
    return frame.iloc[keep].copy().reset_index(drop=True)


def select_labels(
    frame: pd.DataFrame, top_k: int | None, min_support: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support: dict[str, int] = {}
    for labels in frame["labels"]:
        for label in set(labels):
            support[label] = support.get(label, 0) + 1
    stats = pd.DataFrame([{"label": k, "support": v} for k, v in support.items()]).sort_values(
        ["support", "label"], ascending=[False, True]
    )
    eligible = stats[stats["support"] >= min_support]
    selected = set(eligible.head(top_k)["label"] if top_k else eligible["label"])
    result = frame.copy()
    result["labels"] = result["labels"].map(lambda labels: sorted(set(labels) & selected))
    return result[result["labels"].map(bool)].reset_index(drop=True), stats.reset_index(drop=True)

