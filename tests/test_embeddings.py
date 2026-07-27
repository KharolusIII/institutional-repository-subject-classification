import numpy as np

from ir_subject_classification.embeddings import chunk_token_ids, pool_embeddings
from ir_subject_classification.embeddings import DocumentEmbedder


def test_chunking_and_pooling_without_model_download():
    chunks = chunk_token_ids(list(range(10)), max_length=4, overlap=1)
    assert chunks == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [9]]
    vectors = np.array([[1.0, 3.0], [3.0, 1.0]])
    assert np.allclose(pool_embeddings(vectors, [1, 3], "chunked_mean"), [2, 2])
    assert np.allclose(pool_embeddings(vectors, [1, 3], "chunked_length_weighted_mean"), [2.5, 1.5])


def test_document_embeddings_resume_from_per_document_cache(tmp_path):
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))

        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(map(str, tokens))

        def num_special_tokens_to_add(self, pair=False):
            return 2

    class Model:
        max_seq_length = 6
        tokenizer = Tokenizer()

        def encode(self, texts, **kwargs):
            return np.asarray([[len(text.split()), 1.0] for text in texts])

    embedder = DocumentEmbedder.__new__(DocumentEmbedder)
    embedder.model_name = "fake"
    embedder.mode = "chunked_length_weighted_mean"
    embedder.overlap = 1
    embedder.batch_size = 4
    embedder.document_batch_size = 2
    embedder.cache_dir = tmp_path
    embedder.model = Model()
    embedder.last_statistics = {}
    texts = ["a b c d e f g", "a b"]
    first = embedder.encode(texts)
    embedder._cache_path(texts).unlink()

    class FailingModel(Model):
        def encode(self, texts, **kwargs):
            raise AssertionError("Per-document cache was not reused")

    embedder.model = FailingModel()
    second = embedder.encode(texts)
    assert np.allclose(first, second)
    assert embedder.last_statistics["documents_loaded_from_cache"] == 2

