"""Bitstream/internal-ID to handle mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metadata import parse_handle


def build_txt_index(txt_dir: str | Path) -> pd.DataFrame:
    paths = sorted(Path(txt_dir).glob("*.txt"))
    return pd.DataFrame(
        {"txt_filename": [p.name for p in paths], "file_id": [p.stem for p in paths], "txt_path": [str(p) for p in paths]}
    )


def map_files_to_handles(
    files: pd.DataFrame,
    mapping: pd.DataFrame,
    id_column: str = "internal_id",
    handle_column: str = "handle",
) -> pd.DataFrame:
    if id_column not in mapping or handle_column not in mapping:
        raise ValueError(f"Mapping must contain {id_column!r} and {handle_column!r}")
    lookup = mapping[[id_column, handle_column]].copy()
    lookup[id_column] = lookup[id_column].fillna("").astype(str).str.strip()
    lookup[handle_column] = lookup[handle_column].map(parse_handle)
    lookup = lookup.drop_duplicates(id_column, keep="first")
    result = files.copy()
    result["file_id"] = result["file_id"].fillna("").astype(str).str.strip()
    result = result.merge(lookup, left_on="file_id", right_on=id_column, how="left").drop(columns=[id_column])
    return result.rename(columns={handle_column: "handle"})


def mapping_coverage(mapped: pd.DataFrame) -> pd.DataFrame:
    total = len(mapped)
    n_mapped = int(mapped["handle"].notna().sum())
    return pd.DataFrame(
        [
            {
                "txt_files_total": total,
                "mapped_txt_files": n_mapped,
                "handles_with_txt": int(mapped["handle"].dropna().nunique()),
                "coverage_percentage": 100 * n_mapped / total if total else 0.0,
            }
        ]
    )

