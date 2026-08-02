"""Audited abstract-language ground truth matching and evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .metadata import normalize_unicode_spaces, parse_handle


def text_fingerprint(value: object) -> str:
    normalized = normalize_unicode_spaces(value).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_abstract_ground_truth(
    path: str | Path, cache_path: str | Path | None = None
) -> tuple[dict[tuple[str, str], dict[str, str]], pd.DataFrame]:
    source = Path(path)
    source_signature = f"{source.stat().st_size}:{source.stat().st_mtime_ns}"
    cache = Path(cache_path) if cache_path else None
    if cache and cache.is_file():
        cached = pd.read_parquet(cache)
        if "source_signature" in cached and cached["source_signature"].eq(source_signature).all():
            lookup = {
                (str(row.handle), str(row.text_hash)): {
                    "language": str(row.ground_truth_language),
                    "source": str(row.ground_truth_source),
                }
                for row in cached.itertuples(index=False)
            }
            return lookup, pd.DataFrame(columns=["handle", "text_hash", "N_languages"])
    frame = pd.read_csv(source, dtype=str, low_memory=False)
    required = {"dc.identifier.uri", "resumen", "Auditoria_Ultima_Entrenamiento"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Language ground truth is missing columns: {sorted(missing)}")
    frame["handle"] = frame["dc.identifier.uri"].map(parse_handle)
    frame["text_hash"] = frame["resumen"].map(text_fingerprint)
    frame["ground_truth_language"] = (
        frame["Auditoria_Ultima_Entrenamiento"].fillna("").str.strip().str.lower()
    )
    catalogued = frame.get("Idiomacatalogado", pd.Series("", index=frame.index)).fillna("").str.lower()
    historical_langid = frame.get("LangID", pd.Series("", index=frame.index)).fillna("").str.lower()
    frame["ground_truth_source"] = "manual_audit"
    agreement = catalogued.ne("") & catalogued.eq(historical_langid) & catalogued.eq(frame["ground_truth_language"])
    frame.loc[agreement, "ground_truth_source"] = "human_langid_agreement"
    valid = frame[
        frame["handle"].notna()
        & frame["ground_truth_language"].ne("")
        & ~frame["ground_truth_language"].isin(["auditar", "unknown", "empty", "und"])
    ].copy()
    conflicts = (
        valid.groupby(["handle", "text_hash"])["ground_truth_language"]
        .nunique()
        .reset_index(name="N_languages")
    )
    conflicts = conflicts[conflicts["N_languages"].gt(1)]
    conflicting_keys = set(zip(conflicts["handle"], conflicts["text_hash"]))
    lookup = {}
    for row in valid.itertuples(index=False):
        key = (str(row.handle), str(row.text_hash))
        if key not in conflicting_keys:
            lookup[key] = {
                "language": str(row.ground_truth_language),
                "source": str(row.ground_truth_source),
            }
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "handle": handle,
                "text_hash": text_hash,
                "ground_truth_language": values["language"],
                "ground_truth_source": values["source"],
                "source_signature": source_signature,
            }
            for (handle, text_hash), values in lookup.items()
        ]
        temporary = cache.with_suffix(".tmp.parquet")
        pd.DataFrame(rows).to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(cache)
    return lookup, conflicts


def language_ground_truth_reports(segments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    evaluated = segments[segments["ground_truth_language"].notna()].copy()
    if evaluated.empty:
        return {
            "summary": pd.DataFrame([{"matched_segments": 0}]),
            "per_language": pd.DataFrame(),
            "confusion": pd.DataFrame(),
        }
    truth = evaluated["ground_truth_language"].astype(str)
    predicted = evaluated["detected_language"].astype(str)
    report = pd.DataFrame(
        classification_report(truth, predicted, output_dict=True, zero_division=0)
    ).T.reset_index(names="language")
    labels = sorted(set(truth) | set(predicted))
    confusion = pd.DataFrame(
        confusion_matrix(truth, predicted, labels=labels),
        index=labels,
        columns=labels,
    ).reset_index(names="ground_truth_language")
    summary = pd.DataFrame(
        [
            {
                "matched_segments": len(evaluated),
                "matched_handles": evaluated["handle"].nunique(),
                "accuracy": accuracy_score(truth, predicted),
                "manual_audit_N": int(evaluated["ground_truth_source"].eq("manual_audit").sum()),
                "human_langid_agreement_N": int(
                    evaluated["ground_truth_source"].eq("human_langid_agreement").sum()
                ),
            }
        ]
    )
    return {"summary": summary, "per_language": report, "confusion": confusion}
