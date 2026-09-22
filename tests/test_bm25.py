import numpy as np
import scipy.sparse as sp

from ir_subject_classification.vectorizers import BM25Transformer, BM25Vectorizer


def test_bm25_fits_train_and_transforms_new_documents():
    vectorizer = BM25Vectorizer(max_features=20, ngram_range=(1, 1))
    train = vectorizer.fit_transform(["alpha alpha beta", "beta gamma"])
    test = vectorizer.transform(["alpha gamma"])
    assert train.shape[0] == 2
    assert test.shape == (1, train.shape[1])
    assert sp.isspmatrix_csr(train)
    assert sp.isspmatrix_csr(test)
    assert np.isfinite(test.data).all()


def test_bm25_matches_manual_positive_idf_and_keeps_training_statistics():
    counts = sp.csr_matrix(
        np.array(
            [
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 3.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )
    transformer = BM25Transformer()
    assert transformer.k1 == 1.5
    assert transformer.b == 0.75

    transformer.fit(counts)
    document_frequency = np.array([2.0, 2.0, 1.0])
    expected_idf = np.log(1.0 + (3.0 - document_frequency + 0.5) / (document_frequency + 0.5))
    np.testing.assert_allclose(transformer.idf_, expected_idf)
    np.testing.assert_allclose(transformer.avgdl_, 8.0 / 3.0)

    fitted_idf = transformer.idf_.copy()
    fitted_avgdl = transformer.avgdl_
    transformed = transformer.transform(sp.csr_matrix([[0.0, 2.0, 1.0]]))

    term_frequencies = np.array([2.0, 1.0])
    denominator = term_frequencies + 1.5 * (1.0 - 0.75 + 0.75 * 3.0 / (8.0 / 3.0))
    expected = np.zeros((1, 3))
    expected[0, [1, 2]] = term_frequencies * (1.5 + 1.0) / denominator * expected_idf[[1, 2]]

    assert sp.isspmatrix_csr(transformed)
    assert np.isfinite(transformed.data).all()
    np.testing.assert_allclose(transformed.toarray(), expected)
    np.testing.assert_array_equal(transformer.idf_, fitted_idf)
    assert transformer.avgdl_ == fitted_avgdl

