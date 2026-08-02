import numpy as np

from ir_subject_classification.finetuning import (
    _add_special_tokens,
    _encode_documents,
    aggregate_document_logits,
    aggregate_weighted_document_logits,
)


def test_special_tokens_fall_back_to_bert_token_ids_for_transformers_5():
    class Tokenizer:
        cls_token_id = 101
        sep_token_id = 102

    assert _add_special_tokens(Tokenizer(), [7, 8]) == [101, 7, 8, 102]


def test_chunk_logits_are_averaged_per_document():
    logits = np.asarray([[1.0, 3.0], [3.0, 1.0], [5.0, 7.0]])
    document_ids = np.asarray([0, 0, 1])
    result = aggregate_document_logits(logits, document_ids, 2)
    assert np.allclose(result, [[2.0, 2.0], [5.0, 7.0]])


def test_chunk_logits_use_hierarchical_weights():
    logits = np.asarray([[0.0, 2.0], [10.0, 0.0], [4.0, 4.0]])
    result = aggregate_weighted_document_logits(
        logits, np.asarray([0, 0, 1]), np.asarray([0.75, 0.25, 1.0]), 2
    )
    assert np.allclose(result, [[2.5, 1.5], [4.0, 4.0]])


def test_chunk_encoding_honours_configured_length_and_uniform_limit():
    class Tokenizer:
        model_max_length = 10**30
        def num_special_tokens_to_add(self, pair=False): return 2
        def encode(self, text, add_special_tokens=False): return list(range(int(text)))
        def build_inputs_with_special_tokens(self, ids): return [-1, *ids, -2]

    encoded = _encode_documents(Tokenizer(), [[("100", 1.0)]], np.asarray([[1, 0]]), 3, 10)
    assert len(encoded.input_ids) == 3
    assert all(len(chunk) <= 10 for chunk in encoded.input_ids)
    assert np.isclose(encoded.document_weights.sum(), 1.0)
