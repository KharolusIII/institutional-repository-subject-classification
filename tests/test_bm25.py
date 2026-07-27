import numpy as np

from ir_subject_classification.vectorizers import BM25Vectorizer


def test_bm25_fits_train_and_transforms_new_documents():
    vectorizer = BM25Vectorizer(max_features=20, ngram_range=(1, 1))
    train = vectorizer.fit_transform(["alpha alpha beta", "beta gamma"])
    test = vectorizer.transform(["alpha gamma"])
    assert train.shape[0] == 2
    assert test.shape == (1, train.shape[1])
    assert np.isfinite(test.data).all()

