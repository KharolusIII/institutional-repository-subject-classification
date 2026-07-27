import pandas as pd

from ir_subject_classification.pipeline import build_feature_text


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
