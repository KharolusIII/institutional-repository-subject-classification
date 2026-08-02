import pandas as pd

from ir_subject_classification.pipeline import construct_dataset


def test_language_suffix_is_metadata_language_and_cell_value_is_document_language(tmp_path):
    metadata = pd.DataFrame(
        {
            "handle": ["10915/1", "10915/2"],
            "dc.description.abstract[es]": ["An English abstract", "Resumen en español"],
            "dc.language[es]": ["en", "es||pt"],
            "sedici2003.idioma[es]": [None, "es"],
            "dc.subject.materia[es]": ["Label A", "Label B"],
        }
    )
    path = tmp_path / "metadata.csv"
    metadata.to_csv(path, index=False)
    config = {
        "data": {
            "metadata_csv": str(path),
            "handle_column": "handle",
            "abstract_columns": ["dc.description.abstract[es]"],
            "keyword_columns": [],
            "target_columns": ["dc.subject.materia[es]"],
            "max_fulltext_chars": 1000,
        }
    }

    dataset, _ = construct_dataset(config)

    first = dataset.set_index("handle").loc["10915/1"]
    second = dataset.set_index("handle").loc["10915/2"]
    assert first["abstract_declared_language"] == "es"
    assert first["fulltext_declared_language"] == "en"
    assert second["fulltext_declared_language"] == "mul"
