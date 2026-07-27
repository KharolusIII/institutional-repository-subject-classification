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


def create_run_directory(base: str | Path, name: str, resume: bool = False) -> tuple[str, Path]:
    base_path = Path(base)
    if resume and base_path.is_dir():
        candidates = sorted(
            (
                path
                for path in base_path.iterdir()
                if path.is_dir()
                and f"_{name}" in path.name
                and (
                    (path / "RUN_INCOMPLETE").exists()
                    or (path / ".incomplete").exists()
                )
            ),
            reverse=True,
        )
        if candidates:
            return candidates[0].name, candidates[0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = 0
    while True:
        run_id = f"{stamp}_{name}" + (f"_{suffix}" if suffix else "")
        path = Path(base) / run_id
        try:
            path.mkdir(parents=True, exist_ok=False)
            (path / "figures").mkdir()
            (path / "RUN_INCOMPLETE").write_text("Run has not completed.\n", encoding="utf-8")
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
    cooccurrence: pd.DataFrame,
    per_label: pd.DataFrame,
    figures_dir: str | Path,
    overall_metrics: dict[str, float] | None = None,
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

    ranked = per_label.sort_values("f1", ascending=True)
    figure, axis = plt.subplots(figsize=(10, max(8, len(ranked) * 0.32)))
    axis.barh(ranked["label"], ranked["f1"], color="#35689a")
    axis.set_xlim(0, 1)
    axis.set_xlabel("F1 score")
    axis.set_ylabel("Subject label")
    axis.set_title("Per-label F1 score")
    figure.tight_layout()
    figure.savefig(figures_dir / "per_label_f1.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, max(8, len(ranked) * 0.32)))
    axis.barh(ranked["label"], ranked["true_positive"], label="True positive", color="#3a923a")
    axis.barh(
        ranked["label"],
        ranked["false_negative"],
        left=ranked["true_positive"],
        label="False negative",
        color="#d9534f",
    )
    axis.barh(
        ranked["label"],
        ranked["false_positive"],
        left=ranked["true_positive"] + ranked["false_negative"],
        label="False positive",
        color="#f0ad4e",
    )
    axis.set_xlabel("Test documents")
    axis.set_ylabel("Subject label")
    axis.set_title("Correct predictions and errors by label")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(figures_dir / "label_error_counts.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(per_label["support_test"], per_label["predicted_support"], alpha=0.8)
    maximum = max(
        float(per_label["support_test"].max()),
        float(per_label["predicted_support"].max()),
        1,
    )
    axis.plot([0, maximum], [0, maximum], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("True test support")
    axis.set_ylabel("Predicted test support")
    axis.set_title("True vs predicted label prevalence")
    figure.tight_layout()
    figure.savefig(figures_dir / "true_vs_predicted_support.png", dpi=180)
    plt.close(figure)

    columns = 5
    rows = int(np.ceil(len(per_label) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, rows * 3))
    for axis, (_, row) in zip(np.asarray(axes).flat, per_label.iterrows()):
        matrix = np.asarray(
            [
                [row["true_negative"], row["false_positive"]],
                [row["false_negative"], row["true_positive"]],
            ]
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted −", "Predicted +"],
            yticklabels=["True −", "True +"],
            ax=axis,
        )
        axis.set_title(str(row["label"]), fontsize=9)
    for axis in np.asarray(axes).flat[len(per_label) :]:
        axis.axis("off")
    figure.suptitle("Per-label confusion matrices", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(figures_dir / "per_label_confusion_matrices.png", dpi=180)
    plt.close(figure)

    if overall_metrics:
        metric_names = ["f1_macro", "f1_micro", "precision_micro", "recall_micro", "subset_accuracy"]
        available = [name for name in metric_names if name in overall_metrics]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(
            [name.replace("_", " ").title() for name in available],
            [overall_metrics[name] for name in available],
            color="#35689a",
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.set_title("Final test metrics")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(figures_dir / "test_metrics_overview.png", dpi=180)
        plt.close(figure)


def write_evaluation_report(
    path: str | Path,
    best_configuration: dict[str, object],
    test_metrics: dict[str, float],
    per_label: pd.DataFrame,
) -> None:
    """Write a concise English Markdown report; label names remain unchanged."""
    best_labels = per_label.sort_values(["f1", "support_test"], ascending=False).head(10)
    worst_labels = per_label.sort_values(["f1", "support_test"], ascending=[True, False]).head(10)

    def table(frame: pd.DataFrame) -> str:
        lines = ["| Subject label | Test support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
        for row in frame.itertuples(index=False):
            lines.append(
                f"| {row.label} | {row.support_test} | {row.precision:.3f} | "
                f"{row.recall:.3f} | {row.f1:.3f} |"
            )
        return "\n".join(lines)

    configuration = ", ".join(
        f"{key}={value}" for key, value in best_configuration.items()
    )
    content = f"""# Final evaluation report

## Selected configuration

{configuration}

## Overall test metrics

| Metric | Value |
|---|---:|
| Macro F1 | {test_metrics['f1_macro']:.4f} |
| Micro F1 | {test_metrics['f1_micro']:.4f} |
| Micro precision | {test_metrics['precision_micro']:.4f} |
| Micro recall | {test_metrics['recall_micro']:.4f} |
| Subset accuracy | {test_metrics['subset_accuracy']:.4f} |
| Hamming loss | {test_metrics['hamming_loss']:.4f} |

## Best classified subject labels

{table(best_labels)}

## Most difficult subject labels

{table(worst_labels)}

## Interpretation note

Per-label results should be interpreted together with test support. Labels with
very small support have high metric uncertainty.
"""
    Path(path).write_text(content, encoding="utf-8")
