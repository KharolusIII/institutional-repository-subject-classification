"""DSpace metadata parsing and schema reporting."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

import pandas as pd

MISSING_TEXT_MARKERS = {
    "no posee",
    "no se posee",
    "no posee resumen",
    "sin resumen",
    "sin abstract",
    "no disponible",
    "not available",
    "does not have",
}

HANDLE_RE = re.compile(r"(?:/handle/|hdl\.handle\.net/)?(\d{1,6}/\d+)", re.I)


def normalize_unicode_spaces(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def parse_handle(value: object) -> str | None:
    text = normalize_unicode_spaces(value)
    match = HANDLE_RE.search(text)
    return match.group(1) if match else None


def strip_keyword_uri(value: object) -> str:
    return normalize_unicode_spaces(value).split("::", 1)[0].strip()


def parse_multivalue(value: object, separator: str = "||", strip_uri: bool = False) -> list[str]:
    text = normalize_unicode_spaces(value)
    if not text:
        return []
    values = []
    for part in text.split(separator):
        item = strip_keyword_uri(part) if strip_uri else normalize_unicode_spaces(part)
        if item and item not in values:
            values.append(item)
    return values


def is_missing_text_marker(value: object) -> bool:
    normalized = normalize_unicode_spaces(value).casefold().strip(" .:;-_")
    return normalized in MISSING_TEXT_MARKERS


def extract_text_segments(
    row: pd.Series, columns: Iterable[str]
) -> tuple[list[str], list[str], int]:
    """Preserve DSpace value boundaries and their metadata-language suffixes."""
    texts: list[str] = []
    languages: list[str] = []
    missing_markers = 0
    seen: set[str] = set()
    for column in columns:
        declared = "und"
        if column.endswith("]") and "[" in column:
            declared = column.rsplit("[", 1)[1][:-1].lower() or "und"
        for text in parse_multivalue(row.get(column, "")):
            if is_missing_text_marker(text):
                missing_markers += 1
                continue
            fingerprint = normalize_unicode_spaces(text).casefold()
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            texts.append(text)
            languages.append(declared)
    return texts, languages, missing_markers


def parse_labels(row: pd.Series, target_columns: Sequence[str]) -> list[str]:
    labels: set[str] = set()
    for column in target_columns:
        if column in row.index:
            labels.update(parse_multivalue(row[column], strip_uri=True))
    return sorted(labels)


def combine_columns(row: pd.Series, columns: Iterable[str], *, keywords: bool = False) -> str:
    values: list[str] = []
    for column in columns:
        if column not in row.index:
            continue
        parts = parse_multivalue(row[column], strip_uri=keywords)
        for part in parts:
            if part and part not in values:
                values.append(part)
    return " ".join(values)


def discover_columns(columns: Sequence[str]) -> dict[str, list[str]]:
    return {
        "abstract": [c for c in columns if "abstract" in c.lower()],
        "keywords": [c for c in columns if "dc.subject" in c.lower()],
        "uri": [
            c
            for c in columns
            if "identifier.uri" in c.lower() or c.lower() == "handle" or c.lower().endswith(".uri")
        ],
    }


def target_schema_report(frame: pd.DataFrame, target_columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in target_columns:
        if column not in frame:
            continue
        counts: dict[str, int] = {}
        for value in frame[column]:
            for label in parse_multivalue(value, strip_uri=True):
                counts[label] = counts.get(label, 0) + 1
        rows.extend({"label": label, "source_field": column, "support": support} for label, support in counts.items())
    return pd.DataFrame(rows, columns=["label", "source_field", "support"]).sort_values(
        ["source_field", "support", "label"], ascending=[True, False, True]
    )

