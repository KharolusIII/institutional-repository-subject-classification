"""Memory-conscious metadata and full-text ingestion."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


def allocate_text_budget(parts: list[str], max_chars: int) -> list[str]:
    """Allocate a handle-level character budget fairly across source files."""
    values = [str(value) for value in parts if str(value).strip()]
    if sum(map(len, values)) <= max_chars:
        return values
    remaining = max_chars
    result = [""] * len(values)
    active = set(range(len(values)))
    while active and remaining > 0:
        share = max(1, remaining // len(active))
        completed = []
        for index in active:
            available = len(values[index]) - len(result[index])
            take = min(share, available, remaining)
            start = len(result[index])
            result[index] += values[index][start : start + take]
            remaining -= take
            if take == available:
                completed.append(index)
            if remaining == 0:
                break
        active.difference_update(completed)
    return result


def read_metadata(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False, usecols=usecols)


def read_fulltext_txt(mapped: pd.DataFrame, handles: Iterable[str], max_chars: int = 100_000) -> pd.DataFrame:
    wanted = set(map(str, handles))
    texts: dict[str, list[str]] = defaultdict(list)
    source_ids: dict[str, list[str]] = defaultdict(list)
    selected = mapped[mapped["handle"].astype(str).isin(wanted)]
    for row in selected.itertuples(index=False):
        try:
            text = Path(row.txt_path).read_text(encoding="utf-8", errors="replace")
        except (OSError, TypeError):
            text = ""
        if text.strip():
            texts[str(row.handle)].append(text)
            source_ids[str(row.handle)].append(str(row.txt_path))
    rows = []
    for handle, parts in texts.items():
        retained = allocate_text_budget(parts, max_chars)
        rows.append({
            "handle": handle,
            "fulltext_documents": retained,
            "fulltext_source_ids": source_ids[handle][: len(retained)],
            "fulltext": "\n\n".join(retained),
            "fulltext_source_characters": sum(map(len, parts)),
            "fulltext_truncated": sum(map(len, retained)) < sum(map(len, parts)),
        })
    return pd.DataFrame(rows)


def read_fulltext_parquet(
    paths: Iterable[str | Path], handles: Iterable[str], max_chars: int = 100_000
) -> pd.DataFrame:
    wanted = set(map(str, handles))
    texts: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        shard = pd.read_parquet(path, columns=["handle", "text"])
        shard = shard[shard["handle"].astype(str).isin(wanted)]
        for handle, text in shard[["handle", "text"]].itertuples(index=False):
            if isinstance(text, str) and text.strip():
                texts[str(handle)].append(text)
    rows = []
    for handle, parts in texts.items():
        retained = allocate_text_budget(parts, max_chars)
        rows.append({
            "handle": handle,
            "fulltext_documents": retained,
            "fulltext_source_ids": [f"parquet:{index}" for index in range(len(retained))],
            "fulltext": "\n\n".join(retained),
            "fulltext_source_characters": sum(map(len, parts)),
            "fulltext_truncated": sum(map(len, retained)) < sum(map(len, parts)),
        })
    return pd.DataFrame(rows)

