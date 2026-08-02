"""End-to-end orchestration and command-line interface."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from .classifiers import create_classifier, prediction_scores
from .config import load_config, save_resolved_config
from .evaluation import language_performance, per_label_evaluation
from .embeddings import DocumentEmbedder
from .language import create_language_detector
from .logging_utils import configure_run_logging
from .metadata import (
    combine_columns,
    discover_columns,
    extract_text_segments,
    normalize_unicode_spaces,
    parse_handle,
    parse_labels,
    target_schema_report,
)
from .metrics import (
    bootstrap_confidence_intervals,
    multilabel_metrics,
    paired_bootstrap_differences,
)
from .preprocessing import preprocess_sparse, preprocess_transformer
from .reporting import (
    create_run_directory,
    dataset_statistics,
    git_commit,
    label_cooccurrence,
    label_statistics,
    language_distribution,
    text_statistics,
    write_environment,
    write_evaluation_report,
    write_figures,
    write_json,
)
from .sampling import multilabel_sample, select_labels
from .splitting import multilabel_train_validation_test_split
from .thresholds import apply_thresholds, optimize_global_threshold, optimize_per_label_thresholds
from .vectorizers import create_sparse_vectorizer

FEATURE_FIELDS = {
    "abstract": ["abstract"],
    "keywords": ["keywords"],
    "fulltext": ["fulltext"],
    "abstract+keywords": ["abstract", "keywords"],
    "abstract+fulltext": ["abstract", "fulltext"],
    "keywords+fulltext": ["keywords", "fulltext"],
    "abstract+keywords+fulltext": ["abstract", "keywords", "fulltext"],
}
LOGGER = logging.getLogger("ir_subject_classification.pipeline")


def _resolve_columns(frame: pd.DataFrame, configured: Any, kind: str) -> list[str]:
    if configured == "auto" or configured is None:
        return discover_columns(list(frame.columns))[kind]
    return [column for column in configured if column in frame.columns]


def construct_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = config["data"]
    if not data.get("metadata_csv"):
        raise ValueError("Set data.metadata_csv in the configuration")
    metadata = pd.read_csv(data["metadata_csv"], dtype=str, low_memory=False)
    schema = target_schema_report(metadata, data["target_columns"])
    abstract_columns = _resolve_columns(metadata, data.get("abstract_columns"), "abstract")
    keyword_columns = _resolve_columns(metadata, data.get("keyword_columns"), "keywords")
    language_columns = [
        column
        for column in metadata.columns
        if column.startswith("dc.language") or column == "sedici2003.idioma[es]"
    ]

    def declared_document_language(row: pd.Series) -> str:
        values: set[str] = set()
        for column in language_columns:
            for value in str(row.get(column, "") or "").split("||"):
                code = value.strip().lower().split("-", 1)[0]
                if code and code not in {"nan", "none", "other", "und"}:
                    values.add(code)
        return next(iter(values)) if len(values) == 1 else ("mul" if values else "und")

    handle_column = data.get("handle_column", "handle")
    if handle_column in metadata:
        handles = metadata[handle_column].map(parse_handle)
    else:
        uri_columns = discover_columns(list(metadata.columns))["uri"]
        handles = metadata[uri_columns].fillna("").agg(" ".join, axis=1).map(parse_handle)
    abstract_parts = metadata.apply(
        lambda row: extract_text_segments(row, abstract_columns), axis=1
    )
    dataset = pd.DataFrame(
        {
            "handle": handles,
            "abstract_segments": abstract_parts.map(lambda value: value[0]),
            "abstract_segment_declared_languages": abstract_parts.map(lambda value: value[1]),
            "abstract_missing_markers": abstract_parts.map(lambda value: value[2]),
            "fulltext_declared_language": metadata.apply(declared_document_language, axis=1),
            "keywords": metadata.apply(lambda row: combine_columns(row, keyword_columns, keywords=True), axis=1),
            "labels": metadata.apply(lambda row: parse_labels(row, data["target_columns"]), axis=1),
        }
    )
    dataset["abstract"] = dataset["abstract_segments"].map("\n\n".join)
    dataset["abstract_segment_records"] = [
        list(zip(texts, languages))
        for texts, languages in zip(
            dataset["abstract_segments"],
            dataset["abstract_segment_declared_languages"],
        )
    ]
    dataset["abstract_declared_language"] = dataset[
        "abstract_segment_declared_languages"
    ].map(lambda values: values[0] if len(set(values)) == 1 and values else ("mul" if values else "und"))
    if "fulltext" in metadata:
        dataset["fulltext"] = metadata["fulltext"].fillna("").astype(str)
    else:
        # Full text is deliberately deferred until label filtering and sampling.
        dataset["fulltext"] = ""
    dataset["fulltext"] = dataset["fulltext"].fillna("").astype(str).str.slice(
        stop=int(data.get("max_fulltext_chars", 100000))
    )
    dataset["fulltext_documents"] = dataset["fulltext"].map(
        lambda value: [value] if str(value).strip() else []
    )
    dataset["fulltext_source_ids"] = dataset["fulltext_documents"].map(
        lambda values: ["inline:0"] if values else []
    )
    dataset = dataset[dataset["handle"].notna() & dataset["labels"].map(bool)].copy()
    dataset = (
        dataset.groupby("handle", as_index=False)
        .agg(
            {
                "abstract": lambda values: " ".join(dict.fromkeys(filter(None, values))),
                "abstract_segment_records": lambda rows: list(
                    dict.fromkeys(record for values in rows for record in values)
                ),
                "abstract_missing_markers": "sum",
                "abstract_declared_language": lambda values: (
                    next(iter(set(values))) if len(set(values)) == 1 else "und"
                ),
                "fulltext_declared_language": lambda values: (
                    next(iter(set(values))) if len(set(values)) == 1 else "mul"
                ),
                "keywords": lambda values: " ".join(dict.fromkeys(filter(None, values))),
                "fulltext": lambda values: "\n\n".join(filter(None, values))[: int(data.get("max_fulltext_chars", 100000))],
                "fulltext_documents": lambda rows: list(
                    dict.fromkeys(text for values in rows for text in values)
                ),
                "fulltext_source_ids": lambda rows: list(
                    dict.fromkeys(source for values in rows for source in values)
                ),
                "labels": lambda values: sorted({label for labels in values for label in labels}),
            }
        )
        .reset_index(drop=True)
    )
    dataset["abstract_segments"] = dataset["abstract_segment_records"].map(
        lambda records: [record[0] for record in records]
    )
    dataset["abstract_segment_declared_languages"] = dataset[
        "abstract_segment_records"
    ].map(lambda records: [record[1] for record in records])
    dataset = dataset.drop(columns=["abstract_segment_records"])
    dataset["abstract"] = dataset["abstract_segments"].map("\n\n".join)
    dataset["abstract_declared_language"] = dataset[
        "abstract_segment_declared_languages"
    ].map(lambda values: values[0] if len(set(values)) == 1 and values else ("mul" if values else "und"))
    return dataset, schema


def prepare_fulltext_mapping(
    config: dict[str, Any], dataset: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame | list[Path] | None]:
    """Filter by cheap TXT availability without reading text content."""
    data = config["data"]
    if data.get("fulltext_format") == "inline" or not data.get("fulltext_source"):
        return dataset, None
    if data.get("fulltext_format") == "gdrive_api":
        from .drive_api import build_or_load_drive_index
        from .mapping import map_files_to_handles

        if not data.get("mapping_csv"):
            raise ValueError("data.mapping_csv is required for Google Drive API fulltext sources")
        index, _ = build_or_load_drive_index(
            data["fulltext_source"],
            data.get("drive_index_cache"),
            bool(data.get("refresh_drive_index", False)),
        )
        mapping = pd.read_csv(data["mapping_csv"], dtype=str, low_memory=False)
        mapped = map_files_to_handles(
            index,
            mapping,
            data.get("mapping_id_column", "internal_id"),
            data.get("mapping_handle_column", "handle"),
        )
        available = set(mapped["handle"].dropna().astype(str))
        return dataset[dataset["handle"].astype(str).isin(available)].reset_index(drop=True), mapped
    if data.get("fulltext_format") == "parquet":
        source = Path(data["fulltext_source"])
        paths = sorted(source.glob("*.parquet")) if source.is_dir() else [source]
        available: set[str] = set()
        for path in paths:
            handles = pd.read_parquet(path, columns=["handle"])
            available.update(handles["handle"].dropna().astype(str))
        return dataset[dataset["handle"].astype(str).isin(available)].reset_index(drop=True), paths
    if data.get("fulltext_format", "txt") != "txt":
        raise ValueError(f"Unsupported fulltext_format: {data.get('fulltext_format')}")
    from .mapping import build_txt_index, map_files_to_handles

    if not data.get("mapping_csv"):
        raise ValueError("data.mapping_csv is required for TXT fulltext sources")
    mapping = pd.read_csv(data["mapping_csv"], dtype=str, low_memory=False)
    files = build_txt_index(data["fulltext_source"])
    mapped = map_files_to_handles(
        files,
        mapping,
        data.get("mapping_id_column", "internal_id"),
        data.get("mapping_handle_column", "handle"),
    )
    available = set(mapped["handle"].dropna().astype(str))
    return dataset[dataset["handle"].astype(str).isin(available)].reset_index(drop=True), mapped


def attach_selected_fulltext(
    config: dict[str, Any], dataset: pd.DataFrame, mapped: pd.DataFrame | list[Path] | None
) -> pd.DataFrame:
    """Read only the full text required by the already-selected handles."""
    if mapped is None:
        return dataset
    from .ingestion import read_fulltext_parquet, read_fulltext_txt

    materialized_path_value = config["data"].get("materialized_fulltext_parquet")
    materialized_path = Path(materialized_path_value) if materialized_path_value else None
    requested_handles = set(dataset["handle"].astype(str))
    materialized = None
    if materialized_path and materialized_path.exists():
        materialized = pd.read_parquet(materialized_path)
        if config["data"].get("preserve_text_segments", False) and not {
            "fulltext_documents",
            "fulltext_source_ids",
        }.issubset(materialized.columns):
            LOGGER.warning(
                "Ignoring incompatible legacy fulltext Parquet without segment boundaries: %s",
                materialized_path,
            )
            materialized = None
        if materialized is not None:
            materialized["handle"] = materialized["handle"].astype(str)
            available = set(materialized["handle"])
            fulltext = materialized[
                materialized["handle"].isin(requested_handles)
            ].reset_index(drop=True)
            LOGGER.info(
                "Using materialized fulltext Parquet: requested_handles=%d "
                "available_handles=%d missing_handles=%d path=%s",
                len(requested_handles),
                len(set(fulltext["handle"])),
                len(requested_handles - available),
                materialized_path,
            )

    cached = fulltext if materialized is not None else pd.DataFrame()
    cached_handles = set(cached["handle"].astype(str)) if not cached.empty else set()
    missing_handles = requested_handles - cached_handles
    downloaded = pd.DataFrame()
    if missing_handles:
        limit = int(config["data"].get("max_fulltext_chars", 100000))
        if isinstance(mapped, list):
            downloaded = read_fulltext_parquet(mapped, missing_handles, limit)
        elif "drive_file_id" in mapped.columns:
            from .drive_api import read_selected_fulltext

            downloaded = read_selected_fulltext(
                mapped,
                list(missing_handles),
                limit,
                config["data"].get("drive_text_cache"),
            )
        else:
            downloaded = read_fulltext_txt(mapped, missing_handles, limit)
    fulltext = pd.concat([cached, downloaded], ignore_index=True)
    if materialized_path and not downloaded.empty:
        materialized_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = materialized_path.with_suffix(".tmp.parquet")
        consolidated = pd.concat(
            [materialized if materialized is not None else pd.DataFrame(), downloaded],
            ignore_index=True,
        ).drop_duplicates("handle", keep="last")
        consolidated.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, materialized_path)
    result = dataset.drop(
        columns=["fulltext", "fulltext_documents", "fulltext_source_ids"], errors="ignore"
    ).merge(fulltext, on="handle", how="left")
    result["fulltext"] = result["fulltext"].fillna("").astype(str)
    if "fulltext_documents" not in result:
        result["fulltext_documents"] = result["fulltext"].map(
            lambda value: [value] if str(value).strip() else []
        )
    if "fulltext_source_ids" not in result:
        result["fulltext_source_ids"] = result["fulltext_documents"].map(
            lambda values: [f"legacy:{index}" for index in range(len(values))]
        )
    return result[result["fulltext"].str.strip().ne("")].reset_index(drop=True)


def add_language_columns(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    prepared_value = config.get("data", {}).get("materialized_dataset_parquet")
    prepared_path = Path(prepared_value) if prepared_value else None
    if prepared_path and prepared_path.exists():
        prepared = pd.read_parquet(prepared_path)
        prepared["handle"] = prepared["handle"].astype(str)
        requested = frame["handle"].astype(str).tolist()
        if set(requested).issubset(set(prepared["handle"])):
            return (
                prepared.set_index("handle", drop=False)
                .loc[requested]
                .reset_index(drop=True)
            )
    result = frame.copy()
    language_config = config.get("language", {})
    detector = create_language_detector(language_config.get("backend", "heuristic"))
    keyword_min = int(language_config.get("keyword_min_chars", 20))
    segment_count = int(language_config.get("fulltext_segments", 5))
    segment_chars = int(language_config.get("segment_chars", 4000))
    preprocessing_segment_chars = int(
        language_config.get("fulltext_preprocessing_segment_chars", 12000)
    )
    multilingual_min_share = float(language_config.get("multilingual_min_character_share", 0.20))

    def detect_value(text: object, distributed: bool = False) -> tuple[str, float | None]:
        value = str(text or "")
        if not value.strip():
            return "und", None
        if distributed and len(value) > segment_chars and segment_count > 1:
            starts = np.linspace(0, len(value) - segment_chars, segment_count, dtype=int)
            candidates = [detector.detect(value[start : start + segment_chars]) for start in starts]
            counts: dict[str, int] = {}
            for candidate in candidates:
                counts[candidate.language] = counts.get(candidate.language, 0) + 1
            language = max(counts, key=lambda item: (counts[item], item != "und"))
            scores = [
                candidate.score
                for candidate in candidates
                if candidate.language == language and candidate.score is not None
            ]
            return language, float(np.mean(scores)) if scores else None
        prediction = detector.detect(value)
        return prediction.language, prediction.score

    def as_list(value: object) -> list[str]:
        if isinstance(value, (list, tuple, np.ndarray)):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)] if str(value or "").strip() else []

    for field in ("abstract", "fulltext"):
        source_column = "abstract_segments" if field == "abstract" else "fulltext_documents"
        segment_languages = []
        segment_scores = []
        aggregate_languages = []
        aggregate_scores = []
        language_sets = []
        for value in result[source_column]:
            segments = as_list(value)
            predictions = [detect_value(text, distributed=field == "fulltext") for text in segments]
            languages = [prediction[0] for prediction in predictions]
            scores = [prediction[1] for prediction in predictions]
            segment_languages.append(languages)
            segment_scores.append(scores)
            weights: dict[str, int] = {}
            for text, language in zip(segments, languages):
                weights[language] = weights.get(language, 0) + len(text)
            total_weight = sum(weight for language, weight in weights.items() if language != "und")
            significant = sorted(
                language
                for language, weight in weights.items()
                if language != "und" and weight / max(total_weight, 1) >= multilingual_min_share
            )
            dominant = max(weights, key=lambda language: (weights[language], language != "und")) if weights else "und"
            aggregate = "mul" if len(significant) > 1 else dominant
            matching_scores = [score for language, score in predictions if language == aggregate and score is not None]
            language_sets.append(significant)
            aggregate_languages.append(aggregate)
            aggregate_scores.append(float(np.mean(matching_scores)) if matching_scores else None)
        result[f"{field}_segment_languages"] = segment_languages
        result[f"{field}_segment_language_scores"] = segment_scores
        result[f"{field}_detected_language"] = aggregate_languages
        result[f"{field}_language_score"] = aggregate_scores
        result[f"{field}_detected_language_set"] = language_sets

    fulltext_preprocessing_segments = []
    fulltext_preprocessing_languages = []
    for documents in result["fulltext_documents"]:
        units = [
            document[start : start + preprocessing_segment_chars]
            for document in as_list(documents)
            for start in range(0, len(document), preprocessing_segment_chars)
        ]
        predictions = [detect_value(unit) for unit in units]
        fulltext_preprocessing_segments.append(units)
        fulltext_preprocessing_languages.append([language for language, _ in predictions])
    result["fulltext_preprocessing_segments"] = fulltext_preprocessing_segments
    result["fulltext_preprocessing_segment_languages"] = fulltext_preprocessing_languages
    for index, (units, languages) in enumerate(
        zip(fulltext_preprocessing_segments, fulltext_preprocessing_languages)
    ):
        weights: dict[str, int] = {}
        for unit, language in zip(units, languages):
            weights[language] = weights.get(language, 0) + len(unit)
        total = sum(weight for language, weight in weights.items() if language != "und")
        significant = sorted(
            language for language, weight in weights.items()
            if language != "und" and weight / max(total, 1) >= multilingual_min_share
        )
        result.at[index, "fulltext_detected_language_set"] = significant
        if len(significant) > 1:
            result.at[index, "fulltext_detected_language"] = "mul"

    for field in ("keywords",):
        predictions = []
        for text in result[field]:
            if field == "keywords" and len(str(text).strip()) < keyword_min:
                predictions.append(("und", None))
            else:
                predictions.append(detect_value(text, distributed=field == "fulltext"))
        result[f"{field}_detected_language"] = [value[0] for value in predictions]
        result[f"{field}_language_score"] = [value[1] for value in predictions]
    if "abstract_declared_language" not in result:
        result["abstract_declared_language"] = "und"
    if "fulltext_declared_language" not in result:
        result["fulltext_declared_language"] = "und"
    return result


def attach_abstract_language_ground_truth(
    dataset: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .language_ground_truth import load_abstract_ground_truth, text_fingerprint

    path = config.get("language", {}).get("abstract_ground_truth_csv")
    result = dataset.copy()
    if not path or not Path(path).is_file():
        result["abstract_segment_ground_truth_languages"] = result["abstract_segments"].map(
            lambda values: [None] * len(values)
        )
        return result, pd.DataFrame(), pd.DataFrame()
    lookup, conflicts = load_abstract_ground_truth(
        path, config.get("language", {}).get("abstract_ground_truth_cache")
    )
    audit_rows = []
    ground_truth_lists = []
    for row in result.itertuples(index=False):
        ground_truth = []
        declared = list(row.abstract_segment_declared_languages)
        detected = list(row.abstract_segment_languages)
        scores = list(row.abstract_segment_language_scores)
        for index, text in enumerate(row.abstract_segments):
            match = lookup.get((str(row.handle), text_fingerprint(text)))
            language = match["language"] if match else None
            source = match["source"] if match else None
            ground_truth.append(language)
            audit_rows.append(
                {
                    "handle": row.handle,
                    "segment_index": index,
                    "text_hash": text_fingerprint(text),
                    "characters": len(text),
                    "declared_language": declared[index] if index < len(declared) else "und",
                    "detected_language": detected[index] if index < len(detected) else "und",
                    "detection_score": scores[index] if index < len(scores) else None,
                    "ground_truth_language": language,
                    "ground_truth_source": source,
                }
            )
        ground_truth_lists.append(ground_truth)
    result["abstract_segment_ground_truth_languages"] = ground_truth_lists
    return result, pd.DataFrame(audit_rows), conflicts


def build_feature_text(frame: pd.DataFrame, feature_set: str, preprocessing: str) -> list[str]:
    fields = FEATURE_FIELDS[feature_set]
    parts = []
    for field in fields:
        language_column = f"{field}_detected_language"
        field_preprocessing = (
            "normalized"
            if field == "keywords" and preprocessing == "language_stopwords"
            else preprocessing
        )
        if field in {"abstract", "fulltext"}:
            segment_column = (
                "abstract_segments" if field == "abstract" else "fulltext_preprocessing_segments"
            )
            segment_language_column = (
                "abstract_segment_languages"
                if field == "abstract"
                else "fulltext_preprocessing_segment_languages"
            )
            segment_values = (
                frame[segment_column]
                if segment_column in frame
                else frame[field].map(lambda value: [value] if str(value).strip() else [])
            )
            segment_languages = (
                frame[segment_language_column]
                if segment_language_column in frame
                else frame[language_column].map(lambda value: [value])
            )
            values = [
                "\n".join(
                    preprocess_sparse(text, language, field_preprocessing)
                    for text, language in zip(segments, languages)
                )
                for segments, languages in zip(segment_values, segment_languages)
            ]
        else:
            values = [
                preprocess_sparse(text, language, field_preprocessing)
                for text, language in zip(frame[field], frame[language_column])
            ]
        parts.append([f"{field.upper()}: {value}" for value in values])
    return ["\n".join(values) for values in zip(*parts)]


def build_transformer_text(frame: pd.DataFrame, feature_set: str) -> list[str]:
    fields = FEATURE_FIELDS[feature_set]
    return [
        "\n".join(f"{field.upper()}: {preprocess_transformer(row[field])}" for field in fields)
        for _, row in frame.iterrows()
    ]


def build_transformer_segments(
    frame: pd.DataFrame,
    feature_set: str,
    field_weights: dict[str, float] | None = None,
) -> list[list[tuple[str, float]]]:
    """Return independently weighted text units for hierarchical pooling."""
    fields = FEATURE_FIELDS[feature_set]
    configured = field_weights or {"abstract": 0.25, "keywords": 0.15, "fulltext": 0.60}
    active_weights = {field: float(configured.get(field, 1.0)) for field in fields}
    total = sum(active_weights.values()) or 1.0
    active_weights = {field: weight / total for field, weight in active_weights.items()}
    documents: list[list[tuple[str, float]]] = []
    for row in frame.itertuples(index=False):
        units: list[tuple[str, float]] = []
        for field in fields:
            if field == "abstract":
                values = list(row.abstract_segments)
            elif field == "fulltext":
                values = list(row.fulltext_documents)
            else:
                value = str(row.keywords or "").strip()
                values = [value] if value else []
            if not values:
                continue
            unit_weight = active_weights[field] / len(values)
            units.extend(
                (f"{field.upper()}: {preprocess_transformer(value)}", unit_weight)
                for value in values
            )
        weight_total = sum(weight for _, weight in units) or 1.0
        documents.append([(text, weight / weight_total) for text, weight in units])
    return documents


def _default_threshold(score_kind: str) -> float:
    return 0.5 if score_kind == "probabilities" else 0.0


def _calibrate_threshold(
    config: dict[str, Any], y_true: np.ndarray, scores: np.ndarray, score_kind: str
) -> float | np.ndarray:
    mode = config.get("thresholds", {}).get("mode", "default")
    if mode == "default":
        return _default_threshold(score_kind)
    candidates = (
        np.linspace(0.05, 0.95, 19)
        if score_kind == "probabilities"
        else np.unique(
            np.concatenate(
                ([0.0], np.quantile(np.asarray(scores), np.linspace(0.02, 0.98, 49)))
            )
        )
    )
    global_threshold, _ = optimize_global_threshold(y_true, scores, candidates)
    if mode == "per_label_threshold":
        return optimize_per_label_thresholds(
            y_true,
            scores,
            global_threshold,
            int(config["thresholds"].get("minimum_label_support", 20)),
            candidates,
        )
    return global_threshold


def _experiment_key(
    preprocessing: str, feature_set: str, representation: str, classifier: str
) -> tuple[str, str, str, str]:
    return preprocessing, feature_set, representation, classifier


def _validation_family(row: pd.Series) -> str:
    representation = str(row["representation"])
    if str(row["preprocessing"]) == "transformer_finetuned":
        return representation
    if representation.startswith("sbert:"):
        return "sbert_frozen"
    if representation.startswith("labse:"):
        return "labse_frozen"
    return representation


def _write_checkpoint(rows: list[dict[str, object]], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def run_pipeline(config: dict[str, Any]) -> Path:
    run_id, run_dir = create_run_directory(
        config["experiment"].get("output_dir", "outputs"),
        config["experiment"]["name"],
        bool(config["experiment"].get("resume", False)),
    )
    resumed = (run_dir / "config_resolved.yaml").exists()
    logger = configure_run_logging(run_dir, config["experiment"].get("log_level", "INFO"))
    logger.info("%s run_id=%s", "Resuming" if resumed else "Starting", run_id)
    logger.info("Configuration source=%s", config.get("_config_path", "in-memory"))
    logger.info(
        "Execution scale target_n=%s top_k=%s",
        config.get("sampling", {}).get("target_n"),
        config.get("labels", {}).get("top_k"),
    )
    save_resolved_config(config, run_dir / "config_resolved.yaml")
    write_environment(run_dir / "environment.txt")
    (run_dir / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
    timings: list[dict[str, object]] = []

    started = time.perf_counter()
    stage = time.perf_counter()
    dataset, schema = construct_dataset(config)
    logger.info("Dataset constructed: %d handle-level rows before label selection", len(dataset))
    dataset, mapped_fulltext = prepare_fulltext_mapping(config, dataset)
    logger.info("Rows with an available fulltext mapping: %d", len(dataset))
    timings.append({"stage": "ingestion", "seconds": time.perf_counter() - stage})
    schema.to_csv(run_dir / "target_schema_report.csv", index=False)

    labels_config = config.get("labels", {})
    dataset, _ = select_labels(dataset, labels_config.get("top_k"), int(labels_config.get("min_support", 1)))
    logger.info("Rows after label selection: %d", len(dataset))
    target_n = config.get("sampling", {}).get("target_n")
    dataset = multilabel_sample(dataset, target_n, int(config["experiment"].get("seed", 42)))
    logger.info("Rows after sampling: %d", len(dataset))
    before_fulltext = len(dataset)
    requested_fulltext_handles = set(dataset["handle"].astype(str))
    dataset = attach_selected_fulltext(config, dataset, mapped_fulltext)
    available_fulltext_handles = set(dataset["handle"].astype(str))
    pd.DataFrame(
        {
            "handle": sorted(
                requested_fulltext_handles - available_fulltext_handles
            )
        }
    ).to_csv(run_dir / "fulltext_missing_handles.csv", index=False)
    logger.info(
        "Fulltext read complete: requested_handles=%d non_empty_handles=%d",
        before_fulltext,
        len(dataset),
    )
    dataset["content_group"] = dataset["fulltext"].map(
        lambda value: hashlib.sha256(
            normalize_unicode_spaces(value).casefold().encode("utf-8")
        ).hexdigest()
    )
    duplicate_groups = (
        dataset.groupby("content_group")
        .agg(N_handles=("handle", "size"), handles=("handle", lambda values: "||".join(map(str, values))))
        .reset_index()
    )
    duplicate_groups[duplicate_groups["N_handles"].gt(1)].to_csv(
        run_dir / "exact_content_duplicate_groups.csv", index=False
    )
    execution_stage = str(config["experiment"].get("stage", "all")).lower()
    allowed_stages = {"prepare", "sparse", "sbert", "labse", "finalize", "all"}
    if execution_stage not in allowed_stages:
        raise ValueError(
            f"Unknown execution stage {execution_stage!r}; expected one of {sorted(allowed_stages)}"
        )
    logger.info("Execution stage=%s", execution_stage)
    if "fulltext_source_characters" in dataset:
        truncated = dataset["fulltext_truncated"].astype("boolean").fillna(False)
        pd.DataFrame(
            [
                {
                    "documents": len(dataset),
                    "max_fulltext_characters": int(
                        config["data"].get("max_fulltext_chars", 100000)
                    ),
                    "source_characters_total": int(
                        dataset["fulltext_source_characters"].fillna(0).sum()
                    ),
                    "represented_characters_total": int(
                        dataset["fulltext"].str.len().sum()
                    ),
                    "represented_character_percentage": float(
                        100
                        * dataset["fulltext"].str.len().sum()
                        / max(dataset["fulltext_source_characters"].fillna(0).sum(), 1)
                    ),
                    "truncated_documents": int(truncated.sum()),
                    "truncated_document_percentage": float(
                        100 * truncated.mean()
                    ),
                }
            ]
        ).to_csv(run_dir / "fulltext_coverage.csv", index=False)

    stage = time.perf_counter()
    dataset = add_language_columns(dataset, config)
    dataset, abstract_language_segments, ground_truth_conflicts = attach_abstract_language_ground_truth(
        dataset, config
    )
    logger.info(
        "Language identification complete: abstract=%s fulltext=%s",
        dataset["abstract_detected_language"].value_counts().to_dict(),
        dataset["fulltext_detected_language"].value_counts().to_dict(),
    )
    timings.append({"stage": "language_detection", "seconds": time.perf_counter() - stage})
    if not abstract_language_segments.empty:
        from .language_ground_truth import language_ground_truth_reports

        abstract_language_segments.to_csv(
            run_dir / "abstract_language_segment_audit.csv", index=False
        )
        reports = language_ground_truth_reports(abstract_language_segments)
        reports["summary"].to_csv(run_dir / "abstract_language_ground_truth_summary.csv", index=False)
        reports["per_language"].to_csv(run_dir / "abstract_language_ground_truth_per_language.csv", index=False)
        reports["confusion"].to_csv(run_dir / "abstract_language_ground_truth_confusion.csv", index=False)
        ground_truth_conflicts.to_csv(
            run_dir / "abstract_language_ground_truth_conflicts.csv", index=False
        )
    for field in ("abstract", "fulltext", "keywords"):
        language_distribution(dataset[f"{field}_detected_language"]).to_csv(
            run_dir / f"{field}_language_distribution.csv", index=False
        )
    agreement_rows = []
    for field in ("abstract", "fulltext"):
        declared_column = f"{field}_declared_language"
        detected_column = f"{field}_detected_language"
        grouped = (
            dataset.groupby([declared_column, detected_column], dropna=False)
            .size()
            .reset_index(name="N")
        )
        for row in grouped.itertuples(index=False):
            declared, detected, count = row
            agreement_rows.append(
                {
                    "field": field,
                    "declaration_source": (
                        "metadata_field_suffix"
                        if field == "abstract"
                        else "dc.language_or_legacy_value"
                    ),
                    "declared_language": declared,
                    "detected_language": detected,
                    "N": count,
                    "agreement": declared == detected,
                }
            )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(run_dir / "language_agreement.csv", index=False)
    agreement_summary = []
    for field in ("abstract", "fulltext"):
        field_rows = agreement[agreement["field"].eq(field)]
        auditable = field_rows[
            ~field_rows["declared_language"].isin(["und", "mul"])
            & field_rows["detected_language"].ne("und")
        ]
        total = int(auditable["N"].sum())
        matches = int(auditable.loc[auditable["agreement"], "N"].sum())
        agreement_summary.append(
            {
                "field": field,
                "auditable_N": total,
                "agreement_N": matches,
                "disagreement_N": total - matches,
                "agreement_rate": matches / total if total else np.nan,
            }
        )
    pd.DataFrame(agreement_summary).to_csv(
        run_dir / "language_agreement_summary.csv", index=False
    )
    language_subject = dataset.explode("labels").groupby(["abstract_detected_language", "labels"]).size().reset_index(name="N")
    language_subject.to_csv(run_dir / "language_by_subject.csv", index=False)
    pd.DataFrame(
        [
            {
                "N_items": len(dataset),
                "N_abstract_segments": int(dataset["abstract_segments"].map(len).sum()),
                "N_fulltext_files": int(dataset["fulltext_documents"].map(len).sum()),
                "items_without_abstract": int(dataset["abstract_segments"].map(len).eq(0).sum()),
                "items_with_multiple_abstracts": int(dataset["abstract_segments"].map(len).gt(1).sum()),
                "items_with_multiple_fulltexts": int(dataset["fulltext_documents"].map(len).gt(1).sum()),
                "abstract_missing_markers_removed": int(dataset["abstract_missing_markers"].sum()),
                "multilingual_abstract_items": int(
                    dataset["abstract_detected_language_set"].map(lambda values: len(values) > 1).sum()
                ),
                "multilingual_fulltext_items": int(
                    dataset["fulltext_detected_language_set"].map(lambda values: len(values) > 1).sum()
                ),
            }
        ]
    ).to_csv(run_dir / "text_unit_statistics.csv", index=False)

    dataset_statistics(dataset["labels"].tolist()).to_csv(run_dir / "dataset_statistics.csv", index=False)
    label_statistics(dataset["labels"].tolist()).to_csv(run_dir / "label_statistics.csv", index=False)
    text_statistics(dataset).to_csv(run_dir / "text_statistics.csv", index=False)
    co_matrix, co_pairs = label_cooccurrence(dataset["labels"].tolist())
    co_matrix.to_csv(run_dir / "label_cooccurrence_matrix.csv", index=False)
    co_pairs.to_csv(run_dir / "label_pair_statistics.csv", index=False)

    split_config = config["split"]
    dataset, coverage = multilabel_train_validation_test_split(
        dataset,
        validation_size=float(split_config["validation_size"]),
        test_size=float(split_config["test_size"]),
        calibration_size=float(split_config.get("calibration_size", 0.0)),
        seed=int(config["experiment"].get("seed", 42)),
        max_tries=int(split_config.get("max_tries", 40)),
        group_column="content_group" if split_config.get("group_by_content", False) else None,
    )
    logger.info("Split sizes: %s", dataset["split"].value_counts().to_dict())
    logger.info("Labels present in every split: %d/%d", int(coverage["present_in_all_splits"].sum()), len(coverage))
    coverage.to_csv(run_dir / "label_coverage_by_split.csv", index=False)
    split_export = dataset
    if not config.get("artifacts", {}).get("include_text_in_dataset_splits", True):
        split_export = dataset.drop(
            columns=[
                "abstract",
                "abstract_segments",
                "keywords",
                "fulltext",
                "fulltext_documents",
                "fulltext_preprocessing_segments",
            ],
            errors="ignore",
        )
    split_export.to_csv(run_dir / "dataset_splits.csv", index=False)
    try:
        prepared_value = config.get("data", {}).get("materialized_dataset_parquet")
        prepared_path = (
            Path(prepared_value)
            if prepared_value
            else run_dir / "dataset_prepared.parquet"
        )
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = prepared_path.with_suffix(".tmp.parquet")
        dataset.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, prepared_path)
        if prepared_path != run_dir / "dataset_prepared.parquet":
            pd.DataFrame(
                [{"materialized_dataset_parquet": str(prepared_path)}]
            ).to_csv(run_dir / "dataset_materialization.csv", index=False)
    except ImportError:
        logger.warning("PyArrow is unavailable; dataset_prepared.parquet was not written")

    if execution_stage == "prepare":
        timings.append({"stage": "prepare_total", "seconds": time.perf_counter() - started})
        pd.DataFrame(timings).to_csv(run_dir / "timing_prepare.csv", index=False)
        logger.info(
            "Preparation stage completed; rerun with stage=sparse, sbert, labse, or all"
        )
        return run_dir

    mlb = MultiLabelBinarizer()
    y_all = mlb.fit_transform(dataset["labels"])
    train_idx = np.flatnonzero(dataset["split"].eq("train"))
    validation_idx = np.flatnonzero(dataset["split"].eq("validation"))
    calibration_idx = np.flatnonzero(dataset["split"].eq("calibration"))
    if not len(calibration_idx):
        calibration_idx = validation_idx
    test_idx = np.flatnonzero(dataset["split"].eq("test"))
    sparse_representations = [
        item
        for item in config["representations"]["enabled"]
        if item in {"bow", "tfidf", "bm25"}
        and execution_stage in {"sparse", "all"}
    ]
    modes = config.get("preprocessing", {}).get("modes")
    if modes is None:
        modes = [config.get("preprocessing", {}).get("mode", "raw")]
    validation_checkpoint = run_dir / "results_validation_checkpoint.csv"
    if validation_checkpoint.exists():
        validation_rows = pd.read_csv(validation_checkpoint).to_dict("records")
        logger.info("Loaded %d completed validation combinations", len(validation_rows))
    else:
        validation_rows: list[dict[str, object]] = []
    completed = {
        _experiment_key(
            str(row["preprocessing"]),
            str(row["feature_set"]),
            str(row["representation"]),
            str(row["classifier"]),
        )
        for row in validation_rows
    }

    for mode in modes:
        logger.info("Starting sparse preprocessing mode=%s", mode)
        for feature_set in config["features"]["sets"]:
            stage = time.perf_counter()
            texts = build_feature_text(dataset, feature_set, mode)
            timings.append({"stage": "preprocessing", "feature_set": feature_set, "preprocessing": mode, "seconds": time.perf_counter() - stage})
            for representation in sparse_representations:
                pending_classifiers = [
                    name
                    for name in config["classifiers"]["enabled"]
                    if _experiment_key(mode, feature_set, representation, name) not in completed
                ]
                if not pending_classifiers:
                    logger.info(
                        "Checkpoint hit representation=%s feature_set=%s preprocessing=%s",
                        representation,
                        feature_set,
                        mode,
                    )
                    continue
                logger.info(
                    "Fitting representation=%s feature_set=%s preprocessing=%s",
                    representation,
                    feature_set,
                    mode,
                )
                params = config["representations"]
                bm25 = params.get("bm25", {})
                vectorizer = create_sparse_vectorizer(
                    representation,
                    max_features=params.get("max_features"),
                    min_df=params.get("min_df", 1),
                    max_df=params.get("max_df", 1.0),
                    ngram_range=params.get("ngram_range", [1, 2]),
                    lowercase=params.get("lowercase", True),
                    k1=bm25.get("k1", 1.5),
                    b=bm25.get("b", 0.75),
                )
                stage = time.perf_counter()
                x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
                timings.append({"stage": "representation_fit", "feature_set": feature_set, "representation": representation, "seconds": time.perf_counter() - stage})
                stage = time.perf_counter()
                x_validation = vectorizer.transform([texts[i] for i in validation_idx])
                x_calibration = vectorizer.transform([texts[i] for i in calibration_idx])
                timings.append({"stage": "representation_transform", "feature_set": feature_set, "representation": representation, "seconds": time.perf_counter() - stage})
                for classifier_name in pending_classifiers:
                    classifier = create_classifier(
                        classifier_name,
                        config["classifiers"].get(classifier_name, {}),
                        int(config["experiment"].get("seed", 42)),
                    )
                    stage = time.perf_counter()
                    classifier.fit(x_train, y_all[train_idx])
                    fit_time = time.perf_counter() - stage
                    stage = time.perf_counter()
                    calibration_scores, score_kind = prediction_scores(classifier, x_calibration)
                    threshold = _calibrate_threshold(
                        config, y_all[calibration_idx], calibration_scores, score_kind
                    )
                    scores, _ = prediction_scores(classifier, x_validation)
                    prediction = apply_thresholds(scores, threshold)
                    prediction_time = time.perf_counter() - stage
                    metrics = multilabel_metrics(y_all[validation_idx], prediction, scores)
                    validation_rows.append(
                        {
                            "run_id": run_id,
                            "preprocessing": mode,
                            "feature_set": feature_set,
                            "representation": representation,
                            "classifier": classifier_name,
                            "score_kind": score_kind,
                            "threshold": json.dumps(
                                threshold.tolist() if isinstance(threshold, np.ndarray) else float(threshold)
                            ),
                            **metrics,
                        }
                    )
                    completed.add(_experiment_key(mode, feature_set, representation, classifier_name))
                    _write_checkpoint(validation_rows, validation_checkpoint)
                    logger.info(
                        "Validation result repr=%s feature=%s preprocessing=%s classifier=%s f1_macro=%.6f f1_micro=%.6f",
                        representation,
                        feature_set,
                        mode,
                        classifier_name,
                        metrics["f1_macro"],
                        metrics["f1_micro"],
                    )
                    timings.extend(
                        [
                            {"stage": "classifier_fit", "feature_set": feature_set, "representation": representation, "classifier": classifier_name, "seconds": fit_time},
                            {"stage": "prediction", "feature_set": feature_set, "representation": representation, "classifier": classifier_name, "seconds": prediction_time},
                        ]
                    )

    dense_representations = [
        item
        for item in config["representations"]["enabled"]
        if item in {"sbert", "labse"}
        and execution_stage in {item, "all"}
    ]
    embedding_config = config["representations"].get("embeddings", {})
    dense_modes = embedding_config.get("modes") or [embedding_config.get("mode", "legacy_truncated")]
    for feature_set in config["features"]["sets"]:
        segmented_texts = build_transformer_segments(
            dataset, feature_set, config.get("features", {}).get("field_weights")
        )
        for representation in dense_representations:
            model_name = embedding_config.get(
                f"{representation}_model",
                "distiluse-base-multilingual-cased-v1"
                if representation == "sbert"
                else "sentence-transformers/LaBSE",
            )
            for embedding_mode in dense_modes:
                representation_name = f"{representation}:{embedding_mode}"
                pending_classifiers = [
                    name
                    for name in config["classifiers"]["enabled"]
                    if _experiment_key(
                        "transformer_minimal", feature_set, representation_name, name
                    )
                    not in completed
                ]
                if not pending_classifiers:
                    logger.info(
                        "Checkpoint hit representation=%s feature_set=%s",
                        representation_name,
                        feature_set,
                    )
                    continue
                configured_embedding_cache = embedding_config.get("cache_dir")
                embedding_cache = (
                    Path(configured_embedding_cache)
                    if configured_embedding_cache
                    else run_dir / "embedding_cache"
                )
                embedder = DocumentEmbedder(
                    model_name=model_name,
                    mode=embedding_mode,
                    overlap=int(embedding_config.get("overlap", 32)),
                    batch_size=int(embedding_config.get("batch_size", 32)),
                    document_batch_size=int(embedding_config.get("document_batch_size", 32)),
                    cache_dir=embedding_cache if embedding_config.get("cache", True) else None,
                )
                logger.info(
                    "Encoding dense representation=%s mode=%s feature_set=%s model=%s",
                    representation,
                    embedding_mode,
                    feature_set,
                    model_name,
                )
                stage = time.perf_counter()
                coverage_rows = []
                encoded_partitions = {}
                for split_name, indices in (
                    ("train", train_idx),
                    ("calibration", calibration_idx),
                    ("validation", validation_idx),
                ):
                    encoded_partitions[split_name] = embedder.encode_segmented(
                        [segmented_texts[i] for i in indices]
                    )
                    coverage_rows.append(
                        {
                            "partition": split_name,
                            "feature_set": feature_set,
                            "representation": representation_name,
                            **embedder.last_statistics,
                        }
                    )
                x_train = encoded_partitions["train"]
                x_calibration = encoded_partitions["calibration"]
                x_validation = encoded_partitions["validation"]
                pd.DataFrame(coverage_rows).to_csv(
                    run_dir / f"embedding_coverage_{representation}_{embedding_mode}_{feature_set.replace('+', '_')}.csv",
                    index=False,
                )
                timings.append(
                    {
                        "stage": "representation_transform",
                        "feature_set": feature_set,
                        "representation": representation_name,
                        "seconds": time.perf_counter() - stage,
                    }
                )
                for classifier_name in pending_classifiers:
                    classifier = create_classifier(
                        classifier_name,
                        config["classifiers"].get(classifier_name, {}),
                        int(config["experiment"].get("seed", 42)),
                    )
                    stage = time.perf_counter()
                    classifier.fit(x_train, y_all[train_idx])
                    fit_time = time.perf_counter() - stage
                    stage = time.perf_counter()
                    calibration_scores, score_kind = prediction_scores(classifier, x_calibration)
                    threshold = _calibrate_threshold(
                        config, y_all[calibration_idx], calibration_scores, score_kind
                    )
                    scores, _ = prediction_scores(classifier, x_validation)
                    prediction = apply_thresholds(scores, threshold)
                    prediction_time = time.perf_counter() - stage
                    metrics = multilabel_metrics(y_all[validation_idx], prediction, scores)
                    validation_rows.append(
                        {
                            "run_id": run_id,
                            "preprocessing": "transformer_minimal",
                            "feature_set": feature_set,
                            "representation": representation_name,
                            "classifier": classifier_name,
                            "score_kind": score_kind,
                            "threshold": json.dumps(
                                threshold.tolist() if isinstance(threshold, np.ndarray) else float(threshold)
                            ),
                            **metrics,
                        }
                    )
                    completed.add(
                        _experiment_key(
                            "transformer_minimal", feature_set, representation_name, classifier_name
                        )
                    )
                    _write_checkpoint(validation_rows, validation_checkpoint)
                    logger.info(
                        "Validation result repr=%s feature=%s classifier=%s f1_macro=%.6f f1_micro=%.6f",
                        representation_name,
                        feature_set,
                        classifier_name,
                        metrics["f1_macro"],
                        metrics["f1_micro"],
                    )
                    timings.extend(
                        [
                            {
                                "stage": "classifier_fit",
                                "feature_set": feature_set,
                                "representation": representation_name,
                                "classifier": classifier_name,
                                "seconds": fit_time,
                            },
                            {
                                "stage": "prediction",
                                "feature_set": feature_set,
                                "representation": representation_name,
                                "classifier": classifier_name,
                                "seconds": prediction_time,
                            },
                        ]
                    )
    if config.get("finetuning", {}).get("enabled", False) and execution_stage in {"all", "finalize"}:
        from .finetuning import run_finetuning

        finetuning_dir = run_finetuning(dataset, list(mlb.classes_), config, run_dir)
        finetuning_validation = pd.read_csv(finetuning_dir / "results_validation.csv")
        for row in finetuning_validation.to_dict("records"):
            thresholds = np.load(finetuning_dir / str(row["model"]) / "thresholds.npy")
            validation_rows.append(
                {
                    "run_id": run_id,
                    "preprocessing": "transformer_finetuned",
                    "feature_set": row["feature_set"],
                    "representation": row["model"],
                    "classifier": "transformer_head",
                    "score_kind": "decision_function",
                    "threshold": json.dumps(thresholds.tolist()),
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"model", "model_name", "feature_set"}
                    },
                }
            )
    validation = pd.DataFrame(validation_rows).sort_values("f1_macro", ascending=False)
    validation.to_csv(run_dir / "results_validation.csv", index=False)
    if validation.empty:
        raise ValueError("No completed validation experiment is available.")
    if execution_stage not in {"finalize", "all"}:
        timings.append(
            {
                "stage": f"{execution_stage}_total",
                "seconds": time.perf_counter() - started,
            }
        )
        pd.DataFrame(timings).to_csv(
            run_dir / f"timing_{execution_stage}.csv", index=False
        )
        logger.info(
            "Stage %s completed with %d total validation combinations; "
            "RUN_INCOMPLETE is retained for the next stage",
            execution_stage,
            len(validation),
        )
        return run_dir
    expected = {
        _experiment_key(mode, feature_set, representation, classifier)
        for mode in modes
        for feature_set in config["features"]["sets"]
        for representation in config["representations"]["enabled"]
        if representation in {"bow", "tfidf", "bm25"}
        for classifier in config["classifiers"]["enabled"]
    }
    expected.update(
        {
            _experiment_key(
                "transformer_minimal",
                feature_set,
                f"{representation}:{embedding_mode}",
                classifier,
            )
            for feature_set in config["features"]["sets"]
            for representation in config["representations"]["enabled"]
            if representation in {"sbert", "labse"}
            for embedding_mode in dense_modes
            for classifier in config["classifiers"]["enabled"]
        }
    )
    missing = expected - completed
    if missing:
        raise RuntimeError(
            f"Cannot finalize: {len(missing)} of {len(expected)} validation "
            "combinations are still missing. Complete the sparse, sbert, and "
            "labse stages first."
        )

    validation = validation.copy()
    validation["evaluation_family"] = validation.apply(_validation_family, axis=1)
    validation.to_csv(run_dir / "results_validation.csv", index=False)
    best = validation.iloc[0]
    requested_families = config.get("evaluation", {}).get(
        "secondary_test_families",
        ["bow", "tfidf", "bm25", "sbert_frozen", "labse_frozen", "sbert_finetuned", "labse_finetuned"],
    )
    selected_entries = [
        {
            "selection_role": "global_confirmatory",
            "evaluation_family": str(best.evaluation_family),
            **best.to_dict(),
        }
    ]
    for family in requested_families:
        candidates = validation[validation["evaluation_family"].eq(family)]
        if not candidates.empty:
            selected_entries.append(
                {
                    "selection_role": "family_secondary",
                    "evaluation_family": family,
                    **candidates.iloc[0].to_dict(),
                }
            )
    selected = pd.DataFrame(selected_entries)
    selected.to_csv(run_dir / "selected_test_models_frozen.csv", index=False)
    logger.info(
        "Frozen best validation configuration: preprocessing=%s feature_set=%s representation=%s classifier=%s f1_macro=%.6f",
        best.preprocessing,
        best.feature_set,
        best.representation,
        best.classifier,
        best.f1_macro,
    )
    # Keep validation isolated for model selection. A later production refit can
    # use train+validation only after all paper metrics have been frozen.
    final_train_idx = train_idx

    def evaluate_candidate(candidate: pd.Series) -> tuple[np.ndarray, np.ndarray, float | np.ndarray]:
        if str(candidate.preprocessing) == "transformer_finetuned":
            from .finetuning import evaluate_finetuned_model

            return evaluate_finetuned_model(
                dataset,
                list(mlb.classes_),
                config,
                run_dir,
                str(candidate.representation),
            )
        if str(candidate.preprocessing) == "transformer_minimal":
            final_texts = build_transformer_segments(
                dataset,
                str(candidate.feature_set),
                config.get("features", {}).get("field_weights"),
            )
            representation, embedding_mode = str(candidate.representation).split(":", 1)
            model_name = embedding_config.get(
                f"{representation}_model",
                "distiluse-base-multilingual-cased-v1"
                if representation == "sbert"
                else "sentence-transformers/LaBSE",
            )
            configured_embedding_cache = embedding_config.get("cache_dir")
            embedding_cache = Path(configured_embedding_cache) if configured_embedding_cache else run_dir / "embedding_cache"
            vectorizer = DocumentEmbedder(
                model_name=model_name,
                mode=embedding_mode,
                overlap=int(embedding_config.get("overlap", 32)),
                batch_size=int(embedding_config.get("batch_size", 32)),
                document_batch_size=int(embedding_config.get("document_batch_size", 32)),
                cache_dir=embedding_cache if embedding_config.get("cache", True) else None,
            )
            x_train = vectorizer.encode_segmented([final_texts[i] for i in final_train_idx])
            x_calibration = vectorizer.encode_segmented([final_texts[i] for i in calibration_idx])
            x_test = vectorizer.encode_segmented([final_texts[i] for i in test_idx])
        else:
            final_texts = build_feature_text(
                dataset, str(candidate.feature_set), str(candidate.preprocessing)
            )
            params = config["representations"]
            bm25 = params.get("bm25", {})
            vectorizer = create_sparse_vectorizer(
                str(candidate.representation),
                max_features=params.get("max_features"),
                min_df=params.get("min_df", 1),
                max_df=params.get("max_df", 1.0),
                ngram_range=params.get("ngram_range", [1, 2]),
                lowercase=params.get("lowercase", True),
                k1=bm25.get("k1", 1.5),
                b=bm25.get("b", 0.75),
            )
            x_train = vectorizer.fit_transform([final_texts[i] for i in final_train_idx])
            x_calibration = vectorizer.transform([final_texts[i] for i in calibration_idx])
            x_test = vectorizer.transform([final_texts[i] for i in test_idx])
        classifier = create_classifier(
            str(candidate.classifier),
            config["classifiers"].get(str(candidate.classifier), {}),
            int(config["experiment"].get("seed", 42)),
        )
        classifier.fit(x_train, y_all[final_train_idx])
        calibration_scores, score_kind = prediction_scores(classifier, x_calibration)
        scores, _ = prediction_scores(classifier, x_test)
        calibrated_threshold = _calibrate_threshold(
            config, y_all[calibration_idx], calibration_scores, score_kind
        )
        return scores, apply_thresholds(scores, calibrated_threshold), calibrated_threshold

    test_scores, test_prediction, threshold = evaluate_candidate(best)
    threshold_mode = config.get("thresholds", {}).get("mode", "default")
    test_metrics = multilabel_metrics(y_all[test_idx], test_prediction, test_scores)
    logger.info(
        "Final isolated test: f1_macro=%.6f f1_micro=%.6f subset_accuracy=%.6f",
        test_metrics["f1_macro"],
        test_metrics["f1_micro"],
        test_metrics["subset_accuracy"],
    )
    pd.DataFrame([{**best[["preprocessing", "feature_set", "representation", "classifier"]].to_dict(), **test_metrics}]).to_csv(
        run_dir / "results_test.csv", index=False
    )
    serialized_threshold = threshold.tolist() if isinstance(threshold, np.ndarray) else float(threshold)
    write_json({"mode": threshold_mode, "labels": list(mlb.classes_), "thresholds": serialized_threshold}, run_dir / "thresholds.json")
    per_label = per_label_evaluation(
        y_all[test_idx],
        test_prediction,
        test_scores,
        list(mlb.classes_),
        y_all[train_idx].sum(axis=0),
        y_all[validation_idx].sum(axis=0),
        threshold,
    )
    per_label.to_csv(run_dir / "per_label_test.csv", index=False)
    comparison_columns = ["preprocessing", "feature_set", "representation", "classifier"]
    selected["configuration_key"] = selected[comparison_columns].astype(str).agg("|".join, axis=1)
    selected["selection_roles"] = selected.groupby("configuration_key")["selection_role"].transform(
        lambda values: "|".join(sorted(set(values)))
    )
    unique_selected = selected.drop_duplicates("configuration_key", keep="first")
    global_key = "|".join(str(best[column]) for column in comparison_columns)
    comparative_rows = []
    comparative_per_label = []
    paired_rows = []
    comparative_predictions = []
    for _, candidate in unique_selected.iterrows():
        key = str(candidate["configuration_key"])
        if key == global_key:
            candidate_scores, candidate_prediction, candidate_threshold = (
                test_scores,
                test_prediction,
                threshold,
            )
        else:
            candidate_scores, candidate_prediction, candidate_threshold = evaluate_candidate(candidate)
        candidate_metrics = multilabel_metrics(
            y_all[test_idx], candidate_prediction, candidate_scores
        )
        comparative_rows.append(
            {
                "selection_roles": candidate["selection_roles"],
                "evaluation_family": candidate["evaluation_family"],
                **{column: candidate[column] for column in comparison_columns},
                "validation_f1_macro": candidate["f1_macro"],
                "validation_f1_micro": candidate["f1_micro"],
                **candidate_metrics,
            }
        )
        family_per_label = per_label_evaluation(
            y_all[test_idx],
            candidate_prediction,
            candidate_scores,
            list(mlb.classes_),
            y_all[train_idx].sum(axis=0),
            y_all[validation_idx].sum(axis=0),
            candidate_threshold,
        )
        family_per_label.insert(0, "evaluation_family", candidate["evaluation_family"])
        family_per_label.insert(1, "selection_roles", candidate["selection_roles"])
        comparative_per_label.append(family_per_label)
        if key != global_key:
            for row in paired_bootstrap_differences(
                y_all[test_idx],
                test_prediction,
                candidate_prediction,
                int(config.get("metrics", {}).get("comparative_bootstrap_resamples", 1000)),
                seed=int(config["experiment"].get("seed", 42)),
            ):
                paired_rows.append(
                    {"evaluation_family": candidate["evaluation_family"], **row}
                )
        for position, handle in enumerate(dataset.iloc[test_idx]["handle"]):
            comparative_predictions.append(
                {
                    "handle": handle,
                    "evaluation_family": candidate["evaluation_family"],
                    "selection_roles": candidate["selection_roles"],
                    "true_labels": json.dumps(
                        list(mlb.classes_[np.flatnonzero(y_all[test_idx][position])]),
                        ensure_ascii=False,
                    ),
                    "predicted_labels": json.dumps(
                        list(mlb.classes_[np.flatnonzero(candidate_prediction[position])]),
                        ensure_ascii=False,
                    ),
                    "scores": json.dumps(candidate_scores[position].tolist()),
                }
            )
    pd.DataFrame(comparative_rows).sort_values(
        "f1_macro", ascending=False
    ).to_csv(run_dir / "results_test_comparative.csv", index=False)
    pd.concat(comparative_per_label, ignore_index=True).to_csv(
        run_dir / "per_label_test_comparative.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        run_dir / "paired_bootstrap_vs_global.csv", index=False
    )
    comparative_predictions_frame = pd.DataFrame(comparative_predictions)
    try:
        comparative_predictions_frame.to_parquet(
            run_dir / "predictions_test_comparative.parquet", index=False
        )
    except ImportError:
        comparative_predictions_frame.to_csv(
            run_dir / "predictions_test_comparative.csv", index=False
        )
    per_label.sort_values(["f1", "support_test"], ascending=[False, False]).assign(
        performance_rank=lambda frame: np.arange(1, len(frame) + 1)
    ).to_csv(run_dir / "subject_performance_ranking.csv", index=False)
    per_label[
        [
            "label",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
            "specificity",
        ]
    ].to_csv(run_dir / "multilabel_confusion_matrices.csv", index=False)
    substitution_counts: dict[tuple[str, str], int] = {}
    for truth_row, prediction_row in zip(y_all[test_idx], test_prediction):
        missed = list(mlb.classes_[np.flatnonzero((truth_row == 1) & (prediction_row == 0))])
        extra = list(mlb.classes_[np.flatnonzero((truth_row == 0) & (prediction_row == 1))])
        for missed_label in missed:
            for predicted_label in extra:
                key = (str(missed_label), str(predicted_label))
                substitution_counts[key] = substitution_counts.get(key, 0) + 1
    pd.DataFrame(
        [
            {
                "missed_true_label": missed,
                "extra_predicted_label": predicted,
                "N_documents": count,
            }
            for (missed, predicted), count in sorted(
                substitution_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        columns=["missed_true_label", "extra_predicted_label", "N_documents"],
    ).to_csv(run_dir / "label_substitution_errors.csv", index=False)
    write_figures(co_matrix, per_label, run_dir / "figures", test_metrics)
    best_configuration = best[
        ["preprocessing", "feature_set", "representation", "classifier"]
    ].to_dict()
    write_evaluation_report(
        run_dir / "evaluation_report.md",
        best_configuration,
        test_metrics,
        per_label,
    )
    comparative_table = pd.DataFrame(comparative_rows).sort_values(
        "f1_macro", ascending=False
    )
    report_lines = [
        "\n## Pre-specified secondary representation-family comparison\n",
        "These test results are secondary analyses. The global confirmatory winner was frozen from validation before any test metric was computed and is not replaced by this ranking.\n",
        "| Role | Family | Representation | Classifier | Validation Macro F1 | Test Macro F1 | Test Micro F1 |\n",
        "|---|---|---|---|---:|---:|---:|\n",
    ]
    for row in comparative_table.itertuples(index=False):
        report_lines.append(
            f"| {row.selection_roles} | {row.evaluation_family} | {row.representation} | "
            f"{row.classifier} | {row.validation_f1_macro:.4f} | {row.f1_macro:.4f} | {row.f1_micro:.4f} |\n"
        )
    with (run_dir / "evaluation_report.md").open("a", encoding="utf-8") as stream:
        stream.writelines(report_lines)
    language_performance_rows = []
    for field in ("abstract", "fulltext"):
        frame = language_performance(
            dataset.iloc[test_idx][f"{field}_detected_language"].reset_index(drop=True),
            y_all[test_idx],
            test_prediction,
            test_scores,
            int(config.get("metrics", {}).get("language_min_support", 30)),
        )
        frame.insert(0, "field", field)
        language_performance_rows.append(frame)
    pd.concat(language_performance_rows, ignore_index=True).to_csv(
        run_dir / "language_performance.csv", index=False
    )
    bootstrap = bootstrap_confidence_intervals(
        y_all[test_idx],
        test_prediction,
        int(config.get("metrics", {}).get("bootstrap_resamples", 1000)),
        seed=int(config["experiment"].get("seed", 42)),
    )
    pd.DataFrame(bootstrap).to_csv(run_dir / "bootstrap_ci.csv", index=False)

    ranked = np.argsort(-test_scores, axis=1)
    predictions = pd.DataFrame(
        {
            "handle": dataset.iloc[test_idx]["handle"].to_numpy(),
            "true_labels": [json.dumps(list(mlb.classes_[np.flatnonzero(row)]), ensure_ascii=False) for row in y_all[test_idx]],
            "predicted_labels": [json.dumps(list(mlb.classes_[np.flatnonzero(row)]), ensure_ascii=False) for row in test_prediction],
            "ranked_labels": [json.dumps(list(mlb.classes_[row]), ensure_ascii=False) for row in ranked],
            "scores": [json.dumps(row.tolist()) for row in test_scores],
        }
    )
    try:
        predictions.to_parquet(run_dir / "predictions_test.parquet", index=False)
    except ImportError:
        predictions.to_csv(run_dir / "predictions_test.csv", index=False)
    total_seconds = time.perf_counter() - started
    timings.append({"stage": "total", "seconds": total_seconds})
    pd.DataFrame(timings).to_csv(run_dir / "timing.csv", index=False)
    documents_per_hour = len(dataset) * 3600 / max(total_seconds, 1)
    pd.DataFrame(
        [
            {
                "observed_documents": len(dataset),
                "observed_seconds": total_seconds,
                "documents_per_hour_linear": documents_per_hour,
                "projected_documents_24h_linear": int(documents_per_hour * 24),
                "recommended_documents_20h_linear": int(documents_per_hour * 20),
                "warning": (
                    "Linear projection from one hardware/runtime profile; validate at a larger scale."
                ),
            }
        ]
    ).to_csv(run_dir / "scaling_estimate.csv", index=False)
    logger.info("Run completed successfully in %.3f seconds; artifacts=%s", total_seconds, run_dir)
    (run_dir / "_SUCCESS").write_text("Run completed successfully.\n", encoding="utf-8")
    (run_dir / "RUN_INCOMPLETE").unlink(missing_ok=True)
    (run_dir / ".incomplete").unlink(missing_ok=True)
    return run_dir


def _set_path(config: dict[str, Any], path: str, value: str | None) -> None:
    if value is not None:
        section, key = path.split(".", 1)
        config[section][key] = value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument("--metadata-csv")
    parser.add_argument("--mapping-csv")
    parser.add_argument("--fulltext-source")
    parser.add_argument("--output-dir")
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    _set_path(config, "data.metadata_csv", arguments.metadata_csv)
    _set_path(config, "data.mapping_csv", arguments.mapping_csv)
    _set_path(config, "data.fulltext_source", arguments.fulltext_source)
    _set_path(config, "experiment.output_dir", arguments.output_dir)
    try:
        run_dir = run_pipeline(config)
    except Exception:
        logging.getLogger("ir_subject_classification").exception("Run failed")
        raise
    print(f"Run completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
