"""Generate the two reviewer-facing workflow figures from the frozen protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#174A7E"
LIGHT_BLUE = "#EAF2F8"
GREEN = "#237A57"
LIGHT_GREEN = "#E9F6F0"
ORANGE = "#B85C16"
LIGHT_ORANGE = "#FFF2E8"
GRAY = "#4B5563"


def box(ax, x, y, width, height, title, body, *, face=LIGHT_BLUE, edge=BLUE):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.6, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=edge)
    ax.text(x + width / 2, y + height * 0.34, body, ha="center", va="center",
            fontsize=9, color=GRAY, linespacing=1.25)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.5, color="#6B7280"))


def finish(fig, ax, output):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dataset_workflow(output: Path):
    fig, ax = plt.subplots(figsize=(14, 5.2))
    positions = [0.02, 0.215, 0.41, 0.605, 0.80]
    width, height, y = 0.16, 0.50, 0.26
    contents = [
        ("Source linkage", "Metadata CSV\nMapping CSV\nFull-text files", LIGHT_BLUE, BLUE),
        ("Text units", "Link by handle\nPreserve abstracts/files\nRemove absence markers", LIGHT_BLUE, BLUE),
        ("Coverage budget", "Fair allocation per item\n≤ 400,000 characters\n83.45% character coverage", LIGHT_ORANGE, ORANGE),
        ("Frozen cohort", "19,710 documents\n37 labels · 24,326 assignments\nGroup exact full-text duplicates", LIGHT_GREEN, GREEN),
        ("Frozen split", "Train 12,836\nCalibration 1,927\nValidation 1,988 · Test 2,959", LIGHT_GREEN, GREEN),
    ]
    for x, (title, body, face, edge) in zip(positions, contents):
        box(ax, x, y, width, height, title, body, face=face, edge=edge)
    for left, right in zip(positions, positions[1:]):
        arrow(ax, (left + width + 0.008, 0.51), (right - 0.008, 0.51))
    ax.text(0.5, 0.90, "Dataset construction and leakage-controlled partitioning",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#111827")
    ax.text(0.5, 0.10,
            "All 37 labels occur in every partition; representations are fitted on train only.",
            ha="center", fontsize=10, color=GRAY)
    finish(fig, ax, output)


def experimental_workflow(output: Path):
    fig, ax = plt.subplots(figsize=(15, 7.0))
    ax.text(0.5, 0.95, "Calibrated multilingual long-document evaluation protocol",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#111827")

    box(ax, 0.02, 0.38, 0.14, 0.34, "Field-aware input",
        "Abstract units\nKeywords\nFull-text units\nIndependent LangID", face=LIGHT_BLUE, edge=BLUE)

    box(ax, 0.22, 0.66, 0.25, 0.22, "Sparse route",
        "raw · normalized · language stopwords\nBoW · TF-IDF · BM25\n7 field combinations · 3 classifiers",
        face=LIGHT_ORANGE, edge=ORANGE)
    box(ax, 0.22, 0.38, 0.25, 0.22, "Fixed embedding route",
        "DistilUSE · LaBSE\nComplete SentenceTransformer pipelines\nOverlapping chunks · hierarchical pooling",
        face=LIGHT_BLUE, edge=BLUE)
    box(ax, 0.22, 0.10, 0.25, 0.22, "Supervised classifier route",
        "DistilUSE / LaBSE Transformer modules\nNew 37-output head · end-to-end BCE\n≤ 8 chunks per text unit · weighted logits",
        face=LIGHT_GREEN, edge=GREEN)

    box(ax, 0.55, 0.53, 0.17, 0.24, "Calibration",
        "Global threshold\nPer-label thresholds\nSupport ≥ 20",
        face=LIGHT_GREEN, edge=GREEN)
    box(ax, 0.55, 0.20, 0.17, 0.24, "Validation",
        "Primary: Macro-F1\nSelect configurations\nand supervised-training epoch",
        face=LIGHT_GREEN, edge=GREEN)

    box(ax, 0.80, 0.53, 0.17, 0.24, "Held-out test",
        "Final selected model\nPreselected family representatives\nNo test-based selection",
        face=LIGHT_BLUE, edge=BLUE)
    box(ax, 0.80, 0.20, 0.17, 0.24, "Reviewer artifacts",
        "Metrics and 95% CIs\nPaired bootstrap\nPer-label tables and figures",
        face=LIGHT_BLUE, edge=BLUE)

    for y in (0.77, 0.49, 0.21):
        arrow(ax, (0.165, 0.55), (0.212, y))
    # Every representation route supplies scores to both calibration and
    # validation; the junction keeps that many-to-many relationship legible.
    ax.plot([0.51, 0.51], [0.21, 0.77], color="#6B7280", linewidth=1.5)
    for y in (0.77, 0.49, 0.21):
        arrow(ax, (0.478, y), (0.51, y))
    arrow(ax, (0.51, 0.64), (0.542, 0.64))
    arrow(ax, (0.51, 0.32), (0.542, 0.32))

    # Frozen thresholds and validation selection jointly define the test fit.
    ax.plot([0.765, 0.765], [0.32, 0.64], color="#6B7280", linewidth=1.5)
    arrow(ax, (0.73, 0.64), (0.765, 0.64))
    arrow(ax, (0.73, 0.32), (0.765, 0.32))
    arrow(ax, (0.765, 0.64), (0.792, 0.64))
    arrow(ax, (0.885, 0.52), (0.885, 0.45))

    ax.text(0.31, 0.035, "233 candidates: 189 sparse · 42 fixed · 2 supervised",
            ha="center", fontsize=9.5, color=GRAY)
    ax.text(0.64, 0.035, "Dedicated calibration split", ha="center", fontsize=9.5, color=GRAY)
    ax.text(0.885, 0.035, "2,959 held-out test items", ha="center", fontsize=9.5, color=GRAY)
    finish(fig, ax, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper_artifacts/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_workflow(args.output_dir / "dataset_construction_workflow.png")
    experimental_workflow(args.output_dir / "experimental_protocol_workflow.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()
