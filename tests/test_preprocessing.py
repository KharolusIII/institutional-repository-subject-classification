from ir_subject_classification.preprocessing import preprocess_sparse, preprocess_transformer


def test_language_aware_stopwords():
    assert preprocess_sparse("el modelo y la prueba", "es", "language_stopwords") == "modelo prueba"
    assert preprocess_sparse("the model and the test", "en", "language_stopwords") == "model"
    assert preprocess_sparse("o modelo e a prova", "pt", "language_stopwords") == "modelo prova"


def test_und_does_not_remove_stopwords():
    assert preprocess_sparse("the model and the test", "und", "language_stopwords") == "the model and the test"


def test_transformer_never_removes_stopwords():
    assert preprocess_transformer("  the model and the test  ") == "the model and the test"

