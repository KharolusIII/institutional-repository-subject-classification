import pandas as pd

from ir_subject_classification.pipeline import attach_selected_fulltext


def test_partial_materialized_parquet_is_authoritative(tmp_path, monkeypatch):
    parquet = tmp_path / "fulltext.parquet"
    materialized = pd.DataFrame(
        {
            "handle": ["one"],
            "fulltext": ["available text"],
            "fulltext_source_characters": [14],
            "fulltext_truncated": [False],
        }
    )
    parquet.touch()
    monkeypatch.setattr(pd, "read_parquet", lambda path: materialized.copy())
    dataset = pd.DataFrame(
        {
            "handle": ["one", "missing"],
            "fulltext": ["", ""],
        }
    )
    mapped = pd.DataFrame(
        {
            "handle": ["one", "missing"],
            "drive_file_id": ["file-one", "file-missing"],
        }
    )
    config = {
        "data": {
            "materialized_fulltext_parquet": str(parquet),
            "max_fulltext_chars": 100_000,
        }
    }

    result = attach_selected_fulltext(config, dataset, mapped)

    assert result["handle"].tolist() == ["one"]
    assert result["fulltext"].tolist() == ["available text"]
