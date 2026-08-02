"""End-to-end orchestration and command-line interface."""

from __future__ import annotations

import argparse
import ast
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
from .metadata import combine_columns, discover_columns, parse_handle, parse_labels, target_schema_report
from .metrics import bootstrap_confidence_intervals, multilabel_metrics
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

    def declared_abstract_language(row: pd.Series) -> str:
        languages = []
        for column in abstract_columns:
            if str(row.get(column, "") or "").strip() and column.endswith("]") and "[" in column:
                languages.append(column.rsplit("[", 1)[1][:-1].lower())
        unique = sorted(set(languages))
        return unique[0] if len(unique) == 1 else "und"

    handle_column = data.get("handle_column", "handle")
    if handle_column in metadata:
        handles = metadata[handle_column].map(parse_handle)
    else:
        uri_columns = discover_columns(list(metadata.columns))["uri"]
        handles = metadata[uri_columns].fillna("").agg(" ".join, axis=1).map(parse_handle)
    dataset = pd.DataFrame(
        {
            "handle": handles,
            "abstract": metadata.apply(lambda row: combine_columns(row, abstract_columns), axis=1),
            "abstract_declared_language": metadata.apply(declared_abstract_language, axis=1),
            "fulltext_declared_language": metadata.apply(declared_document_language, axis=1),
            "keywords": metadata.apply(lambda row: combine_columns(row, keyword_columns, keywords=True), axis=1),
            "labels": metadata.apply(lambda row: parse_labels(row, data["target_columns"]), axis=1),
        }
    )
    if "fulltext" in metadata:
        dataset["fulltext"] = metadata["fulltext"].fillna("").astype(str)
    else:
        # Full text is deliberately deferred until label filtering and sampling.
        dataset["fulltext"] = ""
    dataset["fulltext"] = dataset["fulltext"].fillna("").astype(str).str.slice(
        stop=int(data.get("max_fulltext_chars", 100000))
    )
    dataset = dataset[dataset["handle"].notna() & dataset["labels"].map(bool)].copy()
    dataset = (
        dataset.groupby("handle", as_index=False)
        .agg(
            {
                "abstract": lambda values: " ".join(dict.fromkeys(filter(None, values))),
                "abstract_declared_language": lambda values: (
                    next(iter(set(values))) if len(set(values)) == 1 else "und"
                ),
                "fulltext_declared_language": lambda values: (
                    next(iter(set(values))) if len(set(values)) == 1 else "mul"
                ),
                "keywords": lambda values: " ".join(dict.fromkeys(filter(None, values))),
                "fulltext": lambda values: "\n\n".join(filter(None, values))[: int(data.get("max_fulltext_chars", 100000))],
                "labels": lambda values: sorted({label for labels in values for label in labels}),
            }
        )
        .reset_index(drop=True)
    )
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

    if materialized is not None:
        pass
    elif isinstance(mapped, list):
        fulltext = read_fulltext_parquet(
            mapped,
            dataset["handle"],
            int(config["data"].get("max_fulltext_chars", 100000)),
        )
    elif "drive_file_id" in mapped.columns:
        from .drive_api import read_selected_fulltext

        fulltext = read_selected_fulltext(
            mapped,
            dataset["handle"],
            int(config["data"].get("max_fulltext_chars", 100000)),
            config["data"].get("drive_text_cache"),
        )
    else:
        fulltext = read_fulltext_txt(
            mapped,
            dataset["handle"],
            int(config["data"].get("max_fulltext_chars", 100000)),
        )
    if materialized_path and materialized is None:
        materialized_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = materialized_path.with_suffix(".tmp.parquet")
        fulltext.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, materialized_path)
    result = dataset.drop(columns=["fulltext"], errors="ignore").merge(fulltext, on="handle", how="left")
    result["fulltext"] = result["fulltext"].fillna("").astype(str)
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

    for field in ("abstract", "fulltext", "keywords"):
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
    logger.info(
        "Language identification complete: abstract=%s fulltext=%s",
        dataset["abstract_detected_language"].value_counts().to_dict(),
        dataset["fulltext_detected_language"].value_counts().to_dict(),
    )
    timings.append({"stage": "language_detection", "seconds": time.perf_counter() - stage})
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
    )
    logger.info("Split sizes: %s", dataset["split"].value_counts().to_dict())
    logger.info("Labels present in every split: %d/%d", int(coverage["present_in_all_splits"].sum()), len(coverage))
    coverage.to_csv(run_dir / "label_coverage_by_split.csv", index=False)
    split_export = dataset
    if not config.get("artifacts", {}).get("include_text_in_dataset_splits", True):
        split_export = dataset.drop(
            columns=["abstract", "keywords", "fulltext"], errors="ignore"
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
        texts = build_transformer_text(dataset, feature_set)
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
                x_train = embedder.encode([texts[i] for i in train_idx])
                x_calibration = embedder.encode([texts[i] for i in calibration_idx])
                x_validation = embedder.encode([texts[i] for i in validation_idx])
                embedding_statistics = {
                    "feature_set": feature_set,
                    "representation": representation_name,
                    **embedder.last_statistics,
                }
                pd.DataFrame([embedding_statistics]).to_csv(
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

    best = validation.iloc[0]
    logger.info(
        "Frozen best validation configuration: preprocessing=%s feature_set=%s representation=%s classifier=%s f1_macro=%.6f",
        best.preprocessing,
        best.feature_set,
        best.representation,
        best.classifier,
        best.f1_macro,
    )
    final_train_idx = np.concatenate([train_idx, validation_idx])
    if best.preprocessing == "transformer_minimal":
        final_texts = build_transformer_text(dataset, best.feature_set)
        representation, embedding_mode = str(best.representation).split(":", 1)
        model_name = embedding_config.get(
            f"{representation}_model",
            "distiluse-base-multilingual-cased-v1"
            if representation == "sbert"
            else "sentence-transformers/LaBSE",
        )
        configured_embedding_cache = embedding_config.get("cache_dir")
        embedding_cache = (
            Path(configured_embedding_cache)
            if configured_embedding_cache
            else run_dir / "embedding_cache"
        )
        vectorizer = DocumentEmbedder(
            model_name=model_name,
            mode=embedding_mode,
            overlap=int(embedding_config.get("overlap", 32)),
            batch_size=int(embedding_config.get("batch_size", 32)),
            document_batch_size=int(embedding_config.get("document_batch_size", 32)),
            cache_dir=embedding_cache if embedding_config.get("cache", True) else None,
        )
        x_train = vectorizer.encode([final_texts[i] for i in final_train_idx])
        classifier = create_classifier(
            str(best.classifier),
            config["classifiers"].get(str(best.classifier), {}),
            int(config["experiment"].get("seed", 42)),
        )
        classifier.fit(x_train, y_all[final_train_idx])
        score_kind = str(best.score_kind)
        x_test = vectorizer.encode([final_texts[i] for i in test_idx])
        calibration_text = vectorizer.encode([final_texts[i] for i in calibration_idx])
    else:
        final_texts = build_feature_text(dataset, best.feature_set, best.preprocessing)
        params = config["representations"]
        bm25 = params.get("bm25", {})
        vectorizer = create_sparse_vectorizer(
            str(best.representation),
            max_features=params.get("max_features"),
            min_df=params.get("min_df", 1),
            max_df=params.get("max_df", 1.0),
            ngram_range=params.get("ngram_range", [1, 2]),
            lowercase=params.get("lowercase", True),
            k1=bm25.get("k1", 1.5),
            b=bm25.get("b", 0.75),
        )
        x_train = vectorizer.fit_transform([final_texts[i] for i in final_train_idx])
        classifier = create_classifier(
            str(best.classifier),
            config["classifiers"].get(str(best.classifier), {}),
            int(config["experiment"].get("seed", 42)),
        )
        classifier.fit(x_train, y_all[final_train_idx])
        score_kind = str(best.score_kind)
        x_test = vectorizer.transform([final_texts[i] for i in test_idx])
        calibration_text = vectorizer.transform([final_texts[i] for i in calibration_idx])
    calibration_scores, _ = prediction_scores(classifier, calibration_text)
    test_scores, _ = prediction_scores(classifier, x_test)
    threshold_mode = config.get("thresholds", {}).get("mode", "default")
    threshold = _calibrate_threshold(
        config, y_all[calibration_idx], calibration_scores, score_kind
    )
    test_prediction = apply_thresholds(test_scores, threshold)
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
    if config.get("finetuning", {}).get("enabled", False):
        # Keep the optional torch/transformers stack out of sparse-only runs.
        from .finetuning import run_finetuning

        run_finetuning(dataset, list(mlb.classes_), config, run_dir)
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
