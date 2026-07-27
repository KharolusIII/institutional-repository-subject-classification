import pandas as pd

from ir_subject_classification.metadata import (
    normalize_unicode_spaces,
    parse_handle,
    parse_labels,
    strip_keyword_uri,
)


def test_handle_parsing():
    assert parse_handle("https://sedici.unlp.edu.ar/handle/10915/123") == "10915/123"
    assert parse_handle("hdl.handle.net/10915/456") == "10915/456"
    assert parse_handle("") is None


def test_label_parsing_and_keyword_uri_removal():
    row = pd.Series({"target": "Violencia::http://vocab/1|| Educación ||Violencia"})
    assert parse_labels(row, ["target"]) == ["Educación", "Violencia"]
    assert strip_keyword_uri("Violencia::http://vocab/1") == "Violencia"


def test_unicode_normalization_preserves_accents():
    assert normalize_unicode_spaces("  Café\u00a0  investigación ") == "Café investigación"

