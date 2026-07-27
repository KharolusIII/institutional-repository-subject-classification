"""Memory-conscious metadata and full-text ingestion."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_metadata(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False, usecols=usecols)


def read_fulltext_txt(mapped: pd.DataFrame, handles: Iterable[str], max_chars: int = 100_000) -> pd.DataFrame:
    wanted = set(map(str, handles))
    texts: dict[str, list[str]] = defaultdict(list)
    selected = mapped[mapped["handle"].astype(str).isin(wanted)]
    for row in selected.itertuples(index=False):
        try:
            text = Path(row.txt_path).read_text(encoding="utf-8", errors="replace")
        except (OSError, TypeError):
            text = ""
        if text.strip():
            texts[str(row.handle)].append(text)
    return pd.DataFrame(
        {
            "handle": list(texts),
            "fulltext": ["\n\n".join(parts)[:max_chars] for parts in texts.values()],
        }
    )


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
    return pd.DataFrame(
        {"handle": list(texts), "fulltext": ["\n\n".join(parts)[:max_chars] for parts in texts.values()]}
    )

