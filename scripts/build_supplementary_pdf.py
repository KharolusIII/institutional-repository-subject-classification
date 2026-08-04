"""Build the reviewer-facing PDF from public aggregate artifacts."""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path("paper_artifacts")
TABLES = ROOT / "tables"
V3 = ROOT / "runs" / "v3_main"
AUDIT = ROOT / "runs" / "convergence_audit"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def p(value, style):
    return Paragraph(escape(str(value)), style)


def metric(row, key):
    return f'{float(row[key]):.4f}'


def config(row):
    keys = ("representation", "feature_set", "preprocessing", "classifier")
    return " / ".join(row.get(key, "") for key in keys if row.get(key))


def build(output: Path):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Centered", parent=styles["Normal"], alignment=TA_CENTER))
    styles["Normal"].fontSize = 8
    styles["Normal"].leading = 10
    doc = SimpleDocTemplate(
        str(output), pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
        topMargin=12 * mm, bottomMargin=13 * mm,
        title="CACIC 2026 Supplementary Material",
        author="Institutional Repository Subject Classification project",
    )
    story = []

    def heading(text, level=1):
        story.append(Paragraph(text, styles[f"Heading{level}"]))

    def table(headers, data, widths=None):
        body = [[p(x, styles["Centered"]) for x in headers]]
        body += [[p(x, styles["Normal"]) for x in row] for row in data]
        item = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9EADBA")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.extend([item, Spacer(1, 4 * mm)])

    def figure(filename, caption, width=245 * mm):
        image = Image(str(ROOT / "figures" / filename))
        ratio = image.imageHeight / image.imageWidth
        image.drawWidth = width
        image.drawHeight = min(width * ratio, 155 * mm)
        story.extend([image, p(caption, styles["Centered"]), Spacer(1, 4 * mm)])

    story.append(Paragraph("CACIC 2026 Supplementary Material", styles["Title"]))
    story.append(Paragraph(
        "Beyond Embeddings: Language-Aware Sparse Representations and Fine-Tuned Multilingual Transformers",
        styles["Heading2"],
    ))
    story.append(Paragraph(
        "Aggregate reviewer evidence only. Repository texts, handles, item-level predictions, "
        "split manifests, model weights, credentials, and personal paths are excluded.", styles["Normal"]
    ))

    heading("S1. Corpus and protocol")
    dataset = rows(V3 / "dataset_statistics.csv")[0]
    units = rows(V3 / "text_unit_statistics.csv")[0]
    coverage = rows(V3 / "fulltext_coverage.csv")[0]
    table(["Property", "Value"], [
        ["Documents", dataset["N_documents"]],
        ["Labels / assignments", f'{dataset["N_labels"]} / {dataset["N_assignments"]}'],
        ["Abstract segments / full-text files", f'{units["N_abstract_segments"]} / {units["N_fulltext_files"]}'],
        ["Items with multiple abstracts / full texts", f'{units["items_with_multiple_abstracts"]} / {units["items_with_multiple_fulltexts"]}'],
        ["Represented full-text characters", f'{float(coverage["represented_character_percentage"]):.2f}%'],
        ["Train / calibration / validation / test", "12,836 / 1,927 / 1,988 / 2,959"],
    ], [105 * mm, 80 * mm])
    figure("dataset_construction_workflow.png", "Figure S1. Dataset construction and frozen partitions.")
    story.append(PageBreak())
    figure("experimental_protocol_workflow.png", "Figure S2. Experimental and selection protocol.")

    heading("S2. Complete experiment overview")
    grid = rows(V3 / "results_validation_grid.csv")
    grid.sort(key=lambda row: float(row["f1_macro"]), reverse=True)
    table(["Rank", "Configuration", "Macro-F1", "Micro-F1"], [
        [i, config(row), metric(row, "f1_macro"), metric(row, "f1_micro")]
        for i, row in enumerate(grid[:20], 1)
    ], [12 * mm, 180 * mm, 28 * mm, 28 * mm])
    story.append(Paragraph(
        "Complete 233-row validation table: https://github.com/KharolusIII/"
        "institutional-repository-subject-classification/blob/main/paper_artifacts/runs/v3_main/"
        "results_validation_grid.csv",
        styles["Normal"],
    ))

    heading("S3. Family-level held-out comparison")
    family = rows(TABLES / "family_test_comparison.csv")
    table(["Family", "Selected feature set", "Validation Macro-F1", "Test Macro-F1", "Test Micro-F1"], [
        [row["evaluation_family"], row["feature_set"], f'{float(row["validation_f1_macro"]):.4f}',
         metric(row, "f1_macro"), metric(row, "f1_micro")] for row in family
    ], [38 * mm, 75 * mm, 38 * mm, 34 * mm, 34 * mm])

    heading("S4. Convergence audit")
    audit = rows(AUDIT / "results_validation.csv")
    audit.sort(key=lambda row: float(row["f1_macro"]), reverse=True)
    table(["Configuration", "Macro-F1", "Micro-F1"], [
        [config(row), metric(row, "f1_macro"), metric(row, "f1_micro")] for row in audit
    ], [185 * mm, 31 * mm, 31 * mm])

    final = rows(TABLES / "final_test_metrics.csv")[0]
    ci = {row["metric"]: row for row in rows(AUDIT / "bootstrap_ci.csv")}
    heading("S5. Final held-out result")
    table(["Metric", "Estimate", "95% CI"], [
        ["Macro-F1", metric(final, "f1_macro"), f'[{float(ci["f1_macro"]["ci_lower"]):.4f}, {float(ci["f1_macro"]["ci_upper"]):.4f}]'],
        ["Micro-F1", metric(final, "f1_micro"), f'[{float(ci["f1_micro"]["ci_lower"]):.4f}, {float(ci["f1_micro"]["ci_upper"]):.4f}]'],
        ["Subset accuracy", metric(final, "subset_accuracy"), "—"],
        ["Hamming loss", metric(final, "hamming_loss"), "—"],
        ["Precision@3 / Recall@3", f'{metric(final, "precision_at_3")} / {metric(final, "recall_at_3")}', "—"],
    ], [70 * mm, 60 * mm, 70 * mm])
    figure("test_metrics_overview.png", "Figure S3. Final test metrics.", 205 * mm)

    story.append(PageBreak())
    heading("S6. Fine-tuned transformer validation")
    transformer = rows(TABLES / "transformer_validation.csv")
    table(["Model", "Feature set", "Selected epoch", "Macro-F1", "Micro-F1"], [
        [row["model"], row["feature_set"], row["selected_epoch"], metric(row, "f1_macro"), metric(row, "f1_micro")]
        for row in transformer
    ], [45 * mm, 95 * mm, 32 * mm, 32 * mm, 32 * mm])

    heading("S7. Paired bootstrap comparisons")
    paired = rows(TABLES / "paired_bootstrap_updated.csv")
    table(["Candidate", "Metric", "Candidate − final BM25", "95% CI", "P(candidate > BM25)"], [
        [row["evaluation_family"], row["metric"], f'{float(row["difference_candidate_minus_final_bm25"]):.4f}',
         f'[{float(row["ci_lower"]):.4f}, {float(row["ci_upper"]):.4f}]',
         f'{float(row["probability_candidate_better"]):.4f}'] for row in paired
    ], [45 * mm, 32 * mm, 46 * mm, 62 * mm, 48 * mm])

    heading("S8. Per-subject results")
    labels = rows(TABLES / "per_label_test.csv")
    labels.sort(key=lambda row: float(row["f1"]), reverse=True)
    table(["Subject", "Support", "Precision", "Recall", "F1", "Average precision"], [
        [row["label"], row["support_test"], metric(row, "precision"), metric(row, "recall"),
         metric(row, "f1"), metric(row, "average_precision")] for row in labels
    ], [95 * mm, 25 * mm, 30 * mm, 30 * mm, 30 * mm, 37 * mm])

    for filename, caption in [
        ("per_label_f1.png", "Figure S4. F1 by subject."),
        ("per_label_confusion_matrices.png", "Figure S5. One-vs-rest confusion matrices."),
        ("true_vs_predicted_support.png", "Figure S6. True versus predicted support."),
        ("label_error_counts.png", "Figure S7. False-positive and false-negative counts."),
        ("support_vs_f1.png", "Figure S8. Test support versus subject-level F1."),
        ("label_cooccurrence_heatmap.png", "Figure S9. Label co-occurrence structure."),
    ]:
        story.append(PageBreak())
        figure(filename, caption)

    heading("S9. Reproducibility package")
    story.append(Paragraph(
        "Complete aggregate CSV evidence, figures, source configurations, the executable public notebook, "
        "and SHA-256 manifest are available at https://github.com/KharolusIII/"
        "institutional-repository-subject-classification/tree/main/paper_artifacts", styles["Normal"]
    ))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(12 * mm, 7 * mm, "CACIC 2026 — Supplementary Material")
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 7 * mm, f"Page {document.page}")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    build(ROOT / "CACIC_2026_Supplementary_Material.pdf")
