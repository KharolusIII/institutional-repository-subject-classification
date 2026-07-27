import numpy as np

from ir_subject_classification.embeddings import chunk_token_ids, pool_embeddings


def test_chunking_and_pooling_without_model_download():
    chunks = chunk_token_ids(list(range(10)), max_length=4, overlap=1)
    assert chunks == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [9]]
    vectors = np.array([[1.0, 3.0], [3.0, 1.0]])
    assert np.allclose(pool_embeddings(vectors, [1, 3], "chunked_mean"), [2, 2])
    assert np.allclose(pool_embeddings(vectors, [1, 3], "chunked_length_weighted_mean"), [2.5, 1.5])

