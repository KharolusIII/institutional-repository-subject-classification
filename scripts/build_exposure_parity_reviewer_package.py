"""Build the privacy-safe exposure-parity reviewer evidence package.

The three input directories are private validation run directories.  They are
read only.  The generated package contains aggregate evidence and checksums,
never item identifiers, texts, predictions, model weights, caches, or logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "paper_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "reviewer_response" / "exposure_parity"
DEFAULT_ARCHIVE = ARTIFACT_ROOT / "packages" / "exposure_parity_validation_artifacts_2026-09-15.zip"
EXPECTED_SEEDS = (13, 42, 73)
SPLITS = ("train", "calibration", "validation")
MODELS = ("bm25_matched", "labse_matched")
CANONICAL_LABELS = (
    "Antropolog\u00eda", "Arquitectura", "Astronom\u00eda", "Bellas Artes", "Bibliotecolog\u00eda",
    "Biolog\u00eda", "Ciencias Agrarias", "Ciencias Astron\u00f3micas", "Ciencias Econ\u00f3micas",
    "Ciencias Exactas", "Ciencias Inform\u00e1ticas", "Ciencias Jur\u00eddicas", "Ciencias M\u00e9dicas",
    "Ciencias Naturales", "Ciencias Sociales", "Ciencias Veterinarias",
    "Ciencias de la Educaci\u00f3n", "Comunicaci\u00f3n", "Comunicaci\u00f3n Social", "Educaci\u00f3n",
    "Educaci\u00f3n F\u00edsica", "Filosof\u00eda", "F\u00edsica", "Geograf\u00eda", "Historia", "Humanidades",
    "Ingenier\u00eda", "Letras", "Odontolog\u00eda", "Periodismo", "Psicolog\u00eda", "Qu\u00edmica",
    "Relaciones Internacionales", "Sociolog\u00eda", "Trabajo Social", "Urbanismo", "Zoolog\u00eda",
)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".sha256", ".txt", ".yaml", ".yml"}
PRIVATE_SUFFIXES = {
    ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".parquet",
    ".arrow", ".feather", ".pkl", ".pickle", ".joblib", ".log",
}
PRIVATE_CONTENT = (
    re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/Users/|/home/)[^\s\"'<>]+", re.I),
    re.compile(r"(?:/content/drive/(?:MyDrive|My Drive)/|[A-Za-z]:[\\/]+Mi unidad[\\/]+)[^\s\"'<>]+", re.I),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,})\b"),
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_yaml(path: Path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.casefold() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def semantic_csv_digest(path: Path) -> str:
    """Hash CSV values independent of harmless row/column serialization order."""
    rows = read_rows(path)
    canonical = [sorted(row.items()) for row in rows]
    canonical.sort(key=lambda row: json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    data = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def repair_mojibake(value: str) -> str:
    """Repair the known UTF-8-as-Latin-1 label export without altering clean text."""
    if not any(marker in value for marker in ("Ã", "Â", "â")):
        return value
    try:
        repaired = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return repaired


def canonical_labels(exported: list[str]) -> list[str]:
    """Validate and restore the known labels from a legacy lossy CSV encoding."""
    if len(exported) != len(CANONICAL_LABELS):
        raise ValueError("Expected the frozen, ordered 37-label target schema")
    for index, (raw, expected) in enumerate(zip(exported, CANONICAL_LABELS)):
        repaired = repair_mojibake(str(raw))
        lossy = "".join(character if ord(character) < 128 else "\N{REPLACEMENT CHARACTER}"
                        for character in expected)
        if repaired not in {expected, lossy}:
            raise ValueError(f"Frozen label at index {index} is incompatible with the canonical schema")
    return list(CANONICAL_LABELS)


def readable_path(path: Path) -> Path:
    """Use the Win32 extended prefix when deep Drive paths exceed MAX_PATH."""
    absolute = os.path.abspath(path)
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        absolute = "\\\\?\\" + absolute
    return Path(absolute)


def require_run(path: Path, expected_seed: int) -> dict:
    path = readable_path(path)
    required = [
        "context.json", "environment.json", "resolved_config.yaml",
        "_TRAINING_COMPLETE", "_VALIDATION_COMPLETE",
        "metrics/results_calibration.csv", "metrics/results_validation.csv",
        "metrics/epoch_history.csv", "manifests/selection_frozen.json",
        "manifests/provenance.json",
    ]
    required += [f"exposure_statistics/documents_{split}.csv" for split in SPLITS]
    required += [f"exposure_statistics/summary_{split}.csv" for split in SPLITS]
    for model in MODELS:
        for split in ("calibration", "validation"):
            required += [
                f"predictions/{model}/{split}_truth.npy",
                f"predictions/{model}/{split}_predictions.npy",
            ]
    missing = [relative for relative in required if not (path / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Seed {expected_seed} run lacks required artifacts: {missing}")
    forbidden = [path / "_FINAL_EVALUATION_COMPLETE", path / "metrics/results_test.csv"]
    if any(item.exists() for item in forbidden):
        raise ValueError(f"Seed {expected_seed} is not validation-only; final-test evidence is out of scope")

    config = read_yaml(path / "resolved_config.yaml")
    context = read_json(path / "context.json")
    provenance = read_json(path / "manifests/provenance.json")
    selection = read_json(path / "manifests/selection_frozen.json")
    environment = read_json(path / "environment.json")
    seeds = {
        int(config["experiment"]["seed"]), int(context["seed"]),
        *{int(row["seed"]) for row in read_rows(path / "metrics/results_validation.csv")},
    }
    if seeds != {expected_seed}:
        raise ValueError(f"Expected seed {expected_seed}, found {sorted(seeds)}")
    if config["evaluation"].get("run_final_test") is not False:
        raise ValueError("The reviewer package requires evaluation.run_final_test=false")
    labels = canonical_labels(context["labels"])
    return {
        "path": path, "seed": expected_seed, "config": config, "context": context,
        "provenance": provenance, "selection": selection, "environment": environment,
        "labels": labels,
    }


def validate_shared_context(runs: list[dict]) -> None:
    reference = runs[0]
    common_context = ("split_fingerprint", "dataset_fingerprint", "split_sizes")
    for run in runs[1:]:
        for key in common_context:
            if run["context"][key] != reference["context"][key]:
                raise ValueError(f"Seed runs disagree on {key}")
        if run["labels"] != reference["labels"]:
            raise ValueError("Seed runs disagree on label order")
        for key in ("commit", "source_digest"):
            if run["provenance"][key] != reference["provenance"][key]:
                raise ValueError(f"Seed runs disagree on provenance {key}")
        left = json.loads(json.dumps(reference["config"]))
        right = json.loads(json.dumps(run["config"]))
        for item in (left, right):
            item["experiment"].pop("seed", None)
            item["experiment"].pop("output_root", None)
            for key in ("prepared_parquet", "split_manifest", "split_context", "labels_json"):
                item["data"].pop(key, None)
        if left != right:
            raise ValueError("Seed runs disagree on scientific configuration")


def collect_metrics(runs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_seed: list[dict] = []
    for run in runs:
        for split in ("calibration", "validation"):
            rows = read_rows(run["path"] / "metrics" / f"results_{split}.csv")
            if {row["model"] for row in rows} != set(MODELS):
                raise ValueError(f"Seed {run['seed']} {split} metrics lack the two matched models")
            by_seed.extend(rows)
    dimensions = {"model", "seed", "split", "synthetic"}
    metric_names = [name for name in by_seed[0] if name not in dimensions]
    aggregate = []
    for split in ("calibration", "validation"):
        for model in MODELS:
            selected = [row for row in by_seed if row["split"] == split and row["model"] == model]
            for metric in metric_names:
                values = [float(row[metric]) for row in selected]
                aggregate.append({
                    "split": split,
                    "model": model,
                    "metric": metric,
                    "n_run_instances": len(values),
                    "n_independent_neural_seeds": len(values) if model == "labse_matched" else 0,
                    "mean": statistics.fmean(values),
                    "sample_sd": statistics.stdev(values),
                    "minimum": min(values), "maximum": max(values),
                })
    differences = []
    for run in runs:
        rows = {row["model"]: row for row in by_seed
                if row["split"] == "validation" and int(row["seed"]) == run["seed"]}
        for metric in metric_names:
            differences.append({
                "seed": run["seed"],
                "metric": metric,
                "bm25_minus_labse": (
                    float(rows["bm25_matched"][metric])
                    - float(rows["labse_matched"][metric])
                ),
            })
    return by_seed, aggregate, differences


def collect_epochs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    history, summary = [], []
    for run in runs:
        rows = read_rows(run["path"] / "metrics" / "epoch_history.csv")
        for row in rows:
            history.append({"seed": run["seed"], **row})
        macro = [float(row["validation_f1_macro"]) for row in rows]
        peak_index = int(np.argmax(macro))
        selected_epoch = int(run["selection"]["best_epoch"])
        selected = next(row for row in rows if int(row["epoch"]) == selected_epoch)
        rule = run["config"]["neural"]["early_stopping"]
        summary.append({
            "seed": run["seed"], "epochs_evaluated": len(rows),
            "selected_epoch": selected_epoch,
            "selected_validation_f1_macro": float(selected["validation_f1_macro"]),
            "selected_validation_f1_micro": float(selected["validation_f1_micro"]),
            "numerical_peak_epoch": int(rows[peak_index]["epoch"]),
            "numerical_peak_f1_macro": macro[peak_index],
            "peak_minus_selected_f1_macro": macro[peak_index] - float(selected["validation_f1_macro"]),
            "stopped_after_epoch": int(rows[-1]["epoch"]),
            "patience": int(rule["patience"]), "minimum_delta": float(rule["minimum_delta"]),
        })
    return history, summary


def _binary_per_label(truth: np.ndarray, predictions: np.ndarray, labels: list[str]) -> list[dict]:
    if truth.shape != predictions.shape or truth.ndim != 2 or truth.shape[1] != len(labels):
        raise ValueError("Prediction/truth shape is incompatible with the frozen label schema")
    if not set(np.unique(truth)).issubset({0, 1}) or not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("Expected binary prediction and truth arrays")
    result = []
    truth = truth.astype(bool)
    predictions = predictions.astype(bool)
    for index, label in enumerate(labels):
        y, p = truth[:, index], predictions[:, index]
        tp = int(np.sum(y & p))
        fp = int(np.sum(~y & p))
        fn = int(np.sum(y & ~p))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.append({
            "label_index": index, "label": label, "support": int(np.sum(y)),
            "predicted_positive": int(np.sum(p)), "true_positive": tp,
            "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return result


def collect_per_label(runs: list[dict], validation_metrics: list[dict]) -> tuple[list[dict], list[dict]]:
    by_seed = []
    for run in runs:
        for model in MODELS:
            base = run["path"] / "predictions" / model
            truth = np.load(base / "validation_truth.npy", allow_pickle=False)
            predictions = np.load(base / "validation_predictions.npy", allow_pickle=False)
            rows = _binary_per_label(truth, predictions, run["labels"])
            expected = next(row for row in validation_metrics
                            if row["split"] == "validation" and row["model"] == model
                            and int(row["seed"]) == run["seed"])
            if not math.isclose(statistics.fmean(row["f1"] for row in rows),
                                float(expected["f1_macro"]), rel_tol=0, abs_tol=1e-12):
                raise ValueError("Per-label reconstruction does not reproduce recorded Macro-F1")
            by_seed.extend({"seed": run["seed"], "model": model, **row} for row in rows)
    aggregate = []
    for model in MODELS:
        for index, label in enumerate(runs[0]["labels"]):
            selected = [row for row in by_seed if row["model"] == model and row["label_index"] == index]
            record = {
                "model": model, "label_index": index, "label": label,
                "support": selected[0]["support"],
                "n_run_instances": len(selected),
                "n_independent_neural_seeds": len(selected) if model == "labse_matched" else 0,
            }
            for metric in ("precision", "recall", "f1"):
                values = [float(row[metric]) for row in selected]
                record[f"{metric}_mean"] = statistics.fmean(values)
                record[f"{metric}_sample_sd"] = statistics.stdev(values)
                record[f"{metric}_min"] = min(values)
                record[f"{metric}_max"] = max(values)
            aggregate.append(record)
    return by_seed, aggregate


def collect_thresholds(runs: list[dict]) -> list[dict]:
    result = []
    for run in runs:
        for model in MODELS:
            values = run["selection"]["models"][model]["thresholds"]
            if len(values) != len(run["labels"]):
                raise ValueError("Threshold vector does not match label order")
            result.extend({
                "seed": run["seed"], "model": model, "label_index": index,
                "label": label, "threshold": float(values[index]),
                "selected_on": run["selection"]["thresholds_selected_on"],
            } for index, label in enumerate(run["labels"]))
    return result


def _numeric_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _exposure_record(split: str, rows: list[dict[str, str]]) -> dict:
    original = _numeric_column(rows, "original_source_characters")
    source = _numeric_column(rows, "source_characters")
    observed = _numeric_column(rows, "observed_source_characters")
    total_tokens = _numeric_column(rows, "total_transformer_tokens")
    selected_tokens = _numeric_column(rows, "unique_selected_transformer_tokens")
    selected_chunks = _numeric_column(rows, "selected_chunks")
    text_units = _numeric_column(rows, "text_units")
    represented_units = _numeric_column(rows, "represented_text_units")
    source_words = _numeric_column(rows, "approx_source_whitespace_tokens")
    observed_words = _numeric_column(rows, "approx_observed_whitespace_tokens")
    per_document_fraction = np.divide(observed, source, out=np.zeros_like(observed), where=source != 0)
    return {
        "split": split, "n_documents": len(rows),
        "original_source_characters": int(original.sum()),
        "capped_source_characters": int(source.sum()),
        "selected_source_characters": int(observed.sum()),
        "cap_retained_percentage": 100 * source.sum() / original.sum(),
        "matched_character_coverage_percentage": 100 * observed.sum() / source.sum(),
        "selected_original_character_percentage": 100 * observed.sum() / original.sum(),
        "source_whitespace_tokens": int(source_words.sum()),
        "selected_whitespace_tokens": int(observed_words.sum()),
        "matched_whitespace_coverage_percentage": 100 * observed_words.sum() / source_words.sum(),
        "transformer_content_tokens": int(total_tokens.sum()),
        "selected_unique_transformer_tokens": int(selected_tokens.sum()),
        "matched_transformer_token_coverage_percentage": 100 * selected_tokens.sum() / total_tokens.sum(),
        "selected_chunks": int(selected_chunks.sum()),
        "selected_chunks_mean": float(selected_chunks.mean()),
        "selected_chunks_median": float(np.median(selected_chunks)),
        "selected_chunks_p95": float(np.percentile(selected_chunks, 95)),
        "selected_chunks_max": int(selected_chunks.max()),
        "source_text_units": int(text_units.sum()),
        "represented_text_units": int(represented_units.sum()),
        "documents_with_multiple_text_units": int(np.sum(text_units > 1)),
        "documents_with_more_than_eight_selected_chunks": int(np.sum(selected_chunks > 8)),
        "document_character_fraction_mean": float(per_document_fraction.mean()),
        "document_character_fraction_median": float(np.median(per_document_fraction)),
        "documents_with_full_capped_source_exposure": int(np.sum(observed == source)),
        "documents_truncated_at_400k_source_cap": int(np.sum(original > source)),
        "tokenless_documents": sum(str(row["tokenless_document"]).casefold() == "true" for row in rows),
    }


def collect_exposure(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    reference = runs[0]["path"] / "exposure_statistics"
    for run in runs[1:]:
        for split in SPLITS:
            for stem in ("documents", "summary"):
                left = reference / f"{stem}_{split}.csv"
                right = run["path"] / "exposure_statistics" / f"{stem}_{split}.csv"
                if semantic_csv_digest(left) != semantic_csv_digest(right):
                    raise ValueError(f"Seed exposure artifacts differ for {stem}_{split}.csv")
    split_rows = {split: read_rows(reference / f"documents_{split}.csv") for split in SPLITS}
    totals = [_exposure_record(split, split_rows[split]) for split in SPLITS]
    totals.append(_exposure_record("all_non_test", sum((split_rows[item] for item in SPLITS), [])))
    distributions = []
    for split in SPLITS:
        for row in read_rows(reference / f"summary_{split}.csv"):
            distributions.append({"split": split, **row})
    return totals, distributions


def sanitized_config(run: dict) -> dict:
    config = json.loads(json.dumps(run["config"]))
    config["experiment"]["output_root"] = "${OUTPUT_ROOT}"
    replacements = {
        "prepared_parquet": "${PREPARED_PARQUET}", "split_manifest": "${SPLIT_MANIFEST}",
        "split_context": "${SPLIT_CONTEXT}", "labels_json": "${LABELS_JSON}",
    }
    config["data"].update(replacements)
    return config


def _find_aggregate(rows: list[dict], split: str, model: str, metric: str) -> dict:
    return next(row for row in rows
                if row["split"] == split and row["model"] == model and row["metric"] == metric)


def build_report(metrics: list[dict], aggregates: list[dict], differences: list[dict],
                 convergence: list[dict], exposure: list[dict], provenance: dict) -> str:
    bm_macro = _find_aggregate(aggregates, "validation", "bm25_matched", "f1_macro")
    bm_micro = _find_aggregate(aggregates, "validation", "bm25_matched", "f1_micro")
    la_macro = _find_aggregate(aggregates, "validation", "labse_matched", "f1_macro")
    la_micro = _find_aggregate(aggregates, "validation", "labse_matched", "f1_micro")
    macro_delta = [row["bm25_minus_labse"] for row in differences if row["metric"] == "f1_macro"]
    micro_delta = [row["bm25_minus_labse"] for row in differences if row["metric"] == "f1_micro"]
    total = next(row for row in exposure if row["split"] == "all_non_test")
    rows = []
    for seed in EXPECTED_SEEDS:
        bm = next(row for row in metrics if row["split"] == "validation"
                  and row["model"] == "bm25_matched" and int(row["seed"]) == seed)
        la = next(row for row in metrics if row["split"] == "validation"
                  and row["model"] == "labse_matched" and int(row["seed"]) == seed)
        selected = next(row for row in convergence if row["seed"] == seed)
        rows.append(
            f"| {seed} | {selected['selected_epoch']} | {float(bm['f1_macro']):.6f} | "
            f"{float(bm['f1_micro']):.6f} | {float(la['f1_macro']):.6f} | "
            f"{float(la['f1_micro']):.6f} | {float(bm['f1_macro']) - float(la['f1_macro']):.6f} |"
        )
    table = "\n".join(rows)
    return f"""# Exposure-parity validation report

## Decision for the present revision

The existing experiment is sufficient to answer the reviewer's *content
exposure* objection as a clearly labelled post-hoc robustness analysis. It is
not necessary to add the hierarchical long-document experiment to the current
paper. That experiment asks a different question—whether increasing and
learning the aggregation of long-document evidence improves LaBSE—and is better
reserved for a subsequent publication.

The control does not make BM25 and LaBSE algorithmically identical. It isolates
the point raised by the reviewer: both routes receive evidence from exactly the
same raw source spans. Model-specific tokenization, normalization and inductive
biases necessarily remain different and are stated as such.

## Paste-ready response to the reviewer

> We thank the reviewer and agree that the original comparison did not isolate
> model family from document-content exposure. We therefore added a post-hoc,
> validation-only exposure-parity analysis using the 16,751 non-test records
> from the unchanged 19,710-document, 37-label cohort and frozen four-way
> partition. The LaBSE fast tokenizer first
> defines exact raw-source offsets for up to eight uniformly spaced,
> non-overlapping 256-token sequences (254 content tokens plus [CLS]/[SEP]) per
> original full-text unit. BM25 is then fitted only on those same raw character
> spans, after the established language-aware sparse preprocessing; it receives
> no lexical content outside the Transformer-selected spans. The frozen LangID
> assignments for the original preprocessing windows are reused only to choose
> BM25 stopword lists. Items containing several full-text files retain the
> historical per-unit allocation, so they may contain more than eight chunks in
> total. The LaBSE backbone and its new 37-label head are fine-tuned end-to-end
> for up to 20 epochs. Decision thresholds are selected exclusively on the
> frozen threshold-selection partition (named `calibration` in the
> implementation); no probability or score calibration is performed.
> Checkpoints are selected by validation Macro-F1 under an
> early-stopping rule frozen before the three reported runs (patience=2,
> minimum delta=0.002), and LaBSE is repeated with seeds 13, 42 and 73. No test
> inference or evaluation was performed in this follow-up; historical test
> results from the earlier study had already been inspected. On validation,
> BM25 obtained Macro-F1
> {bm_macro['mean']:.6f} and Micro-F1 {bm_micro['mean']:.6f}; exposure-matched
> LaBSE obtained Macro-F1 {la_macro['mean']:.6f} ± {la_macro['sample_sd']:.6f}
> and Micro-F1 {la_micro['mean']:.6f} ± {la_micro['sample_sd']:.6f} across seeds.
> BM25 led in all three runs, by {abs(statistics.fmean(macro_delta)):.6f} Macro-F1
> and {abs(statistics.fmean(micro_delta)):.6f} Micro-F1 on average. Thus, the
> BM25-versus-LaBSE ranking persists when raw content exposure is held constant,
> although we explicitly limit the conclusion to these two routes, this corpus,
> this split and this post-hoc validation control. The result does not establish
> exposure parity for SBERT or for every Transformer configuration.

## Paste-ready paragraph for the paper

> **Post-hoc exposure-parity analysis.** To examine whether the sparse model's
> advantage was attributable to access to more document content, we constructed
> a validation-only matched-exposure control on the frozen cohort and split.
> Exact raw-source spans were selected through the pinned LaBSE fast tokenizer
> (up to eight uniformly spaced, non-overlapping sequences of 256 model tokens,
> including two special tokens, per original full-text unit). The LaBSE
> sequence classifier consumed those token sequences, while BM25 consumed only
> the corresponding raw character spans after its model-appropriate,
> language-aware preprocessing. Frozen LangID assignments derived for the
> original full-text preprocessing windows selected BM25 stopword lists, but no
> unselected lexical content entered its feature matrix. Across the
> {total['n_documents']:,} training, threshold-selection and validation items,
> the common spans represented
> {total['selected_source_characters']:,} of {total['capped_source_characters']:,}
> capped source characters ({total['matched_character_coverage_percentage']:.2f}%).
> With decision-threshold fitting restricted to the threshold-selection
> partition (named `calibration` in the implementation) and validation-only
> checkpoint selection, BM25 achieved Macro-F1 {bm_macro['mean']:.6f} and Micro-F1
> {bm_micro['mean']:.6f}. Across three LaBSE fine-tuning seeds, the corresponding
> values were {la_macro['mean']:.6f} ± {la_macro['sample_sd']:.6f} and
> {la_micro['mean']:.6f} ± {la_micro['sample_sd']:.6f}, respectively. BM25 led
> for every seed. These findings show that the observed ranking is not explained
> solely by unequal raw-content exposure; they do not establish equivalence of
> the models' internal representations or preprocessing. No test inference or
> evaluation was performed for this post-hoc control.

## Compact table for the paper

| Seed | Selected LaBSE epoch | BM25 Macro-F1 | BM25 Micro-F1 | LaBSE Macro-F1 | LaBSE Micro-F1 | BM25 − LaBSE Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
{table}
| **Mean** | — | **{bm_macro['mean']:.6f}** | **{bm_micro['mean']:.6f}** | **{la_macro['mean']:.6f}** | **{la_micro['mean']:.6f}** | **{statistics.fmean(macro_delta):.6f}** |

Values are validation results. The `±` values reported in the prose are sample
standard deviations across LaBSE seeds. The thresholded BM25 F1 values are
identical in all three run contexts; BM25 was refitted in each context as an
integrity control. No inferential test is claimed from three seeds on one fixed
split.

## Protocol and interpretation audit

- Frozen split: 12,836 train, 1,927 threshold selection (implementation name:
  `calibration`), 1,988 validation and 2,959 test; all 37 labels remain in the
  fixed schema.
- Exposure allocation: historical `per_unit`; a maximum of eight sequences per
  source full-text unit, not a global maximum of eight per repository item.
- Chunk geometry: 256 model tokens, 254 content tokens, no overlap, uniformly
  spaced over each unit's available windows.
- LaBSE: end-to-end fine-tuning with a 37-output chunk classifier and weighted
  mean aggregation. Maximum 20 epochs; early stopping uses Macro-F1, patience 2
  and minimum delta 0.002.
- BM25: the same raw spans only, language-aware stopword preprocessing, train-only
  vocabulary, 100,000 maximum features, and one-vs-rest Linear SVC.
- Language side information: frozen LangID assignments from the original
  full-text preprocessing windows select BM25 stopword lists; language is not
  redetected from short spans, and no unselected lexical content enters BM25.
- Decision thresholds: selected only on the dedicated threshold-selection
  partition (internally named `calibration`). Epochs: selected only on validation.
  No test inference or evaluation was performed in the new analysis; historical
  test results from the earlier study had already been inspected.
- Historical execution source revision: `{provenance['code_commit']}`; pinned LaBSE revision:
  `{provenance['model_revision']}`.

## Scope limitation

This analysis supports a narrow and useful conclusion: within the frozen
validation protocol, BM25's lead over the tested LaBSE route persists when both
models are restricted to identical raw-source spans. It does not establish the
same result for SBERT or all Transformer configurations, nor determine whether
a hierarchical Transformer exposed to substantially more text would outperform
either matched route. The latter is the planned follow-up experiment and should
not be implied by the present table.
"""


def build_readme() -> str:
    return """# Exposure-parity reviewer artifacts

This directory contains the privacy-safe aggregate evidence for the post-hoc
validation experiment that restricts BM25 and fine-tuned LaBSE to identical raw
full-text spans. Start with [`REPORT.md`](REPORT.md) for the interpretation and
paste-ready reviewer/manuscript text.

## Contents

- `tables/metrics_by_seed.csv`: complete threshold-selection (`calibration` in
  the implementation) and validation metrics.
- `tables/metrics_aggregate.csv`: run-instance summaries; LaBSE uncertainty is
  the sample SD across three independent training seeds, while the thresholded
  BM25 F1 values are identical across the three run contexts.
- `tables/paired_validation_differences.csv`: BM25-minus-LaBSE differences.
- `tables/epoch_history.csv` and `tables/convergence_summary.csv`: the complete
  early-stopping record and selected checkpoints.
- `tables/exposure_totals.csv` and `tables/exposure_distributions.csv`: aggregate
  common-span coverage, with no document identifiers.
- `tables/per_label_validation_by_seed.csv` and
  `tables/per_label_validation_aggregate.csv`: 37-label validation summaries.
- `tables/thresholds_by_seed.csv`: decision thresholds selected on the
  internally named `calibration` partition, in the frozen label order.
- `configs/`: path-neutral configurations reconstructed from each resolved run.
- `provenance.json`, `environment.json` and `source_artifact_checksums.csv`:
  curated reproducibility records.
- `MANIFEST.sha256`: byte-level checksums for every other file in this directory.

No test inference or evaluation was performed in this follow-up; historical
test results from the earlier study had already been inspected. No source text,
repository handle, item-level prediction, split membership, NumPy array,
model/checkpoint, cache, log, credential or personal/cloud path is included.

The `code_commit` recorded in `provenance.json` identifies the historical
execution revision. This aggregate evidence release does not claim to
distribute the restricted corpus or the private training workspace.

## Rebuild

From the repository root, point the builder to the three complete private run
directories:

```bash
python scripts/build_exposure_parity_reviewer_package.py \\
  --seed13-dir /path/to/seed13-validation-run \\
  --seed42-dir /path/to/seed42-validation-run \\
  --seed73-dir /path/to/seed73-validation-run \\
  --overwrite
```

The builder validates run compatibility, reconstructs aggregate per-label
metrics in memory, scans its outputs for private payloads, creates the
deterministic ZIP, and refreshes the top-level paper-artifact manifest.
"""


def source_checksums(runs: list[dict]) -> list[dict]:
    relative_files = [
        "context.json", "environment.json", "resolved_config.yaml",
        "metrics/results_calibration.csv", "metrics/results_validation.csv",
        "metrics/epoch_history.csv", "manifests/selection_frozen.json",
        "manifests/provenance.json",
    ]
    relative_files += [f"exposure_statistics/{stem}_{split}.csv"
                       for split in SPLITS for stem in ("documents", "summary")]
    for model in MODELS:
        for split in ("calibration", "validation"):
            relative_files += [f"predictions/{model}/{split}_truth.npy",
                               f"predictions/{model}/{split}_predictions.npy"]
    return [{
        "seed": run["seed"], "source_artifact": relative,
        "sha256": sha256(run["path"] / relative), "copied_into_package": False,
    } for run in runs for relative in relative_files]


def curated_provenance(runs: list[dict]) -> dict:
    first = runs[0]
    return {
        "artifact_schema": "exposure-parity-reviewer-evidence-v1",
        "analysis_scope": "post-hoc validation-only exposure-parity robustness analysis",
        "generated_for_date": "2026-09-15",
        "seeds": list(EXPECTED_SEEDS),
        "code_commit": first["provenance"]["commit"],
        "source_digest": first["provenance"]["source_digest"],
        "dataset_fingerprint": first["context"]["dataset_fingerprint"],
        "split_fingerprint": first["context"]["split_fingerprint"],
        "split_sizes": first["context"]["split_sizes"],
        "labels": first["labels"],
        "run_fingerprints": {str(run["seed"]): run["context"]["fingerprint"] for run in runs},
        "model_name": first["config"]["model"]["name"],
        "model_revision": first["config"]["model"]["revision"],
        "tokenizer_name": first["config"]["model"]["tokenizer_name"],
        "tokenizer_revision": first["config"]["model"]["tokenizer_revision"],
        "thresholds_selected_on": "calibration",
        "checkpoints_selected_on": "validation",
        "followup_test_inference_performed": False,
        "followup_test_evaluation_performed": False,
        "historical_test_results_previously_inspected": True,
        "privacy_boundary": "aggregate evidence only; private source directories are not recorded",
    }


def create_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest:
            entries.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    write_text(manifest, "\n".join(entries))


def audit_package(root: Path) -> None:
    forbidden_columns = {"handle", "text", "abstract", "fulltext", "file_id", "uri", "path"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.casefold() in PRIVATE_SUFFIXES:
            raise ValueError(f"Private payload format generated: {path.relative_to(root)}")
        if path.suffix.casefold() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                header = {name.casefold() for name in next(csv.reader(stream), [])}
            if header & forbidden_columns:
                raise ValueError(f"Item-level column generated: {path.relative_to(root)}")
        if path.suffix.casefold() in TEXT_SUFFIXES:
            content = path.read_text(encoding="utf-8-sig")
            for pattern in PRIVATE_CONTENT:
                if pattern.search(content):
                    raise ValueError(f"Private content detected in {path.relative_to(root)}")


def deterministic_zip(source: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    prefix = archive.stem
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = f"{prefix}/{path.relative_to(source).as_posix()}"
            info = ZipInfo(relative, date_time=(2026, 9, 15, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            bundle.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)


def write_root_manifest() -> None:
    output = ARTIFACT_ROOT / "MANIFEST.sha256"
    entries = []
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if path.is_file() and path != output:
            entries.append(f"{portable_sha256(path)}  {path.relative_to(ARTIFACT_ROOT).as_posix()}")
    write_text(output, "\n".join(entries))


def build(runs: list[dict], output: Path, archive: Path, overwrite: bool = False) -> None:
    validate_shared_context(runs)
    by_seed, aggregate, differences = collect_metrics(runs)
    history, convergence = collect_epochs(runs)
    per_label, per_label_aggregate = collect_per_label(runs, by_seed)
    thresholds = collect_thresholds(runs)
    exposure, exposure_distributions = collect_exposure(runs)
    provenance = curated_provenance(runs)

    output = output.resolve()
    archive = archive.resolve()
    expected_parent = (ARTIFACT_ROOT / "reviewer_response").resolve()
    if output.parent != expected_parent:
        raise ValueError(f"Output must be an immediate child of {expected_parent}")
    if archive.parent != (ARTIFACT_ROOT / "packages").resolve() or archive.suffix.casefold() != ".zip":
        raise ValueError("Archive must be a ZIP immediately under paper_artifacts/packages")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to rebuild it")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".exposure-parity-", dir=output.parent))
    try:
        write_text(staging / "README.md", build_readme())
        write_text(staging / "REPORT.md", build_report(by_seed, aggregate, differences,
                                                        convergence, exposure, provenance))
        write_rows(staging / "tables" / "metrics_by_seed.csv", by_seed[0].keys(), by_seed)
        write_rows(staging / "tables" / "metrics_aggregate.csv", aggregate[0].keys(), aggregate)
        write_rows(staging / "tables" / "paired_validation_differences.csv",
                   differences[0].keys(), differences)
        write_rows(staging / "tables" / "epoch_history.csv", history[0].keys(), history)
        write_rows(staging / "tables" / "convergence_summary.csv", convergence[0].keys(), convergence)
        write_rows(staging / "tables" / "exposure_totals.csv", exposure[0].keys(), exposure)
        write_rows(staging / "tables" / "exposure_distributions.csv",
                   exposure_distributions[0].keys(), exposure_distributions)
        write_rows(staging / "tables" / "per_label_validation_by_seed.csv",
                   per_label[0].keys(), per_label)
        write_rows(staging / "tables" / "per_label_validation_aggregate.csv",
                   per_label_aggregate[0].keys(), per_label_aggregate)
        write_rows(staging / "tables" / "thresholds_by_seed.csv", thresholds[0].keys(), thresholds)
        for run in runs:
            destination = staging / "configs" / f"exposure_matched_seed{run['seed']}.yaml"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(yaml.safe_dump(sanitized_config(run), sort_keys=False,
                                                   allow_unicode=True), encoding="utf-8", newline="\n")
        write_json(staging / "provenance.json", provenance)
        environments = {str(run["seed"]): run["environment"] for run in runs}
        write_json(staging / "environment.json", {"environments_by_seed": environments})
        checksums = source_checksums(runs)
        write_rows(staging / "source_artifact_checksums.csv", checksums[0].keys(), checksums)
        create_manifest(staging)
        audit_package(staging)

        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    if archive.exists() and not overwrite:
        raise FileExistsError(f"Archive already exists: {archive}; pass --overwrite to rebuild it")
    deterministic_zip(output, archive)
    archive_hash = sha256(archive)
    write_text(archive.with_suffix(archive.suffix + ".sha256"),
               f"{archive_hash}  {archive.name}")
    with ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or not names:
            raise ValueError("Generated archive is empty or contains duplicate paths")
        bundle.testzip()
    write_root_manifest()
    print(f"Reviewer artifacts: {output}")
    print(f"Archive: {archive} ({archive.stat().st_size} bytes)")
    print(f"SHA-256: {archive_hash}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed13-dir", type=Path, required=True)
    parser.add_argument("--seed42-dir", type=Path, required=True)
    parser.add_argument("--seed73-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    runs = [require_run(getattr(arguments, f"seed{seed}_dir"), seed) for seed in EXPECTED_SEEDS]
    build(runs, arguments.output, arguments.archive, arguments.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
