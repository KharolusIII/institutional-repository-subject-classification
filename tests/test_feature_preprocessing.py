import pandas as pd

from ir_subject_classification.pipeline import build_feature_text, build_transformer_segments


def test_stopwords_are_field_specific_and_never_removed_from_keywords():
    frame = pd.DataFrame(
        {
            "abstract": ["the model and the test"],
            "abstract_detected_language": ["en"],
            "fulltext": ["el modelo y la prueba"],
            "fulltext_detected_language": ["es"],
            "keywords": ["the model and test"],
            "keywords_detected_language": ["en"],
        }
    )
    text = build_feature_text(
        frame, "abstract+keywords+fulltext", "language_stopwords"
    )[0]
    assert "ABSTRACT: model" in text
    assert "FULLTEXT: modelo prueba" in text
    assert "KEYWORDS: the model and test" in text


def test_hierarchical_units_preserve_files_and_normalize_item_weight():
    frame = pd.DataFrame(
        {
            "abstract_segments": [["Resumen", "Abstract"]],
            "fulltext_documents": [["File one", "File two"]],
            "keywords": ["topic"],
        }
    )
    units = build_transformer_segments(frame, "abstract+keywords+fulltext")[0]
    assert len(units) == 5
    assert abs(sum(weight for _, weight in units) - 1.0) < 1e-9
    fulltext_weights = [weight for text, weight in units if text.startswith("FULLTEXT:")]
    assert fulltext_weights == [0.30, 0.30]
