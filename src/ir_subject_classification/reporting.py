"""Run manifests, statistics, co-occurrence, and safe artifact writing."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def create_run_directory(base: str | Path, name: str) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = 0
    while True:
        run_id = f"{stamp}_{name}" + (f"_{suffix}" if suffix else "")
        path = Path(base) / run_id
        try:
            path.mkdir(parents=True, exist_ok=False)
            (path / "figures").mkdir()
            return run_id, path
        except FileExistsError:
            suffix += 1


def write_environment(path: str | Path) -> None:
    text = f"python={sys.version}\nplatform={platform.platform()}\n"
    try:
        packages = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True
        ).stdout
        text += packages
    except OSError:
        pass
    Path(path).write_text(text, encoding="utf-8")


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() or "unavailable"


def dataset_statistics(labels: list[list[str]]) -> pd.DataFrame:
    counts = np.asarray([len(item) for item in labels], dtype=float)
    assignments = int(counts.sum())
    unique = sorted({label for item in labels for label in item})
    return pd.DataFrame(
        [
            {
                "N_documents": len(labels),
                "N_labels": len(unique),
                "N_assignments": assignments,
                "label_cardinality": float(counts.mean()) if len(counts) else 0,
                "label_density": float(counts.mean() / len(unique)) if unique else 0,
                "labels_per_document_mean": float(counts.mean()) if len(counts) else 0,
                "labels_per_document_median": float(np.median(counts)) if len(counts) else 0,
                "labels_per_document_std": float(counts.std()) if len(counts) else 0,
                "labels_per_document_min": float(counts.min()) if len(counts) else 0,
                "labels_per_document_max": float(counts.max()) if len(counts) else 0,
            }
        ]
    )


def label_statistics(labels: list[list[str]]) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for item in labels:
        for label in set(item):
            counts[label] = counts.get(label, 0) + 1
    return pd.DataFrame([{"label": label, "support": support} for label, support in counts.items()]).sort_values(
        ["support", "label"], ascending=[False, True]
    )


def text_statistics(frame: pd.DataFrame, fields: tuple[str, ...] = ("abstract", "keywords", "fulltext")) -> pd.DataFrame:
    rows = []
    for field in fields:
        text = frame[field].fillna("").astype(str)
        lengths = text.str.len()
        tokens = text.str.split().str.len()
        missing = int((text.str.strip() == "").sum())
        rows.append(
            {
                "field": field,
                "missing_N": missing,
                "missing_percentage": 100 * missing / len(frame) if len(frame) else 0,
                "characters_mean": lengths.mean(),
                "characters_median": lengths.median(),
                "characters_p25": lengths.quantile(0.25),
                "characters_p75": lengths.quantile(0.75),
                "characters_p95": lengths.quantile(0.95),
                "tokens_mean": tokens.mean(),
                "tokens_median": tokens.median(),
                "tokens_p95": tokens.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def language_distribution(series: pd.Series) -> pd.DataFrame:
    counts = series.fillna("und").value_counts(dropna=False)
    return pd.DataFrame(
        {"language": counts.index.astype(str), "N": counts.values, "percentage": 100 * counts.values / counts.sum()}
    )


def label_cooccurrence(labels: list[list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    classes = sorted({label for item in labels for label in item})
    matrix = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    supports = {label: 0 for label in classes}
    for item in labels:
        unique = sorted(set(item))
        for a in unique:
            supports[a] += 1
            for b in unique:
                matrix.loc[a, b] += 1
    pairs = []
    for i, a in enumerate(classes):
        for b in classes[i + 1 :]:
            co = int(matrix.loc[a, b])
            union = supports[a] + supports[b] - co
            pairs.append({"label_a": a, "label_b": b, "cooccurrence": co, "jaccard": co / union if union else 0})
    return matrix.rename_axis("label").reset_index(), pd.DataFrame(pairs)


def write_json(value: object, path: str | Path) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_figures(
    cooccurrence: pd.DataFrame, per_label: pd.DataFrame, figures_dir: str | Path
) -> None:
    """Write optional publication-ready plots when plotting extras are installed."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    matrix = cooccurrence.set_index("label")
    figure, axis = plt.subplots(figsize=(max(8, len(matrix) * 0.3), max(6, len(matrix) * 0.3)))
    sns.heatmap(matrix, cmap="viridis", ax=axis)
    axis.set_title("Label co-occurrence")
    figure.tight_layout()
    figure.savefig(figures_dir / "label_cooccurrence_heatmap.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(per_label["support_test"], per_label["f1"], alpha=0.75)
    axis.set_xscale("log")
    axis.set_xlabel("Test support (log scale)")
    axis.set_ylabel("F1")
    axis.set_title("Label support vs F1")
    figure.tight_layout()
    figure.savefig(figures_dir / "support_vs_f1.png", dpi=180)
    plt.close(figure)
