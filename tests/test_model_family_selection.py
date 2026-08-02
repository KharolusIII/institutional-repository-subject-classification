import pandas as pd

from ir_subject_classification.pipeline import _validation_family


def test_validation_family_distinguishes_frozen_and_finetuned_transformers():
    assert _validation_family(pd.Series({"preprocessing": "raw", "representation": "bm25"})) == "bm25"
    assert _validation_family(
        pd.Series({"preprocessing": "transformer_minimal", "representation": "sbert:chunked_mean"})
    ) == "sbert_frozen"
    assert _validation_family(
        pd.Series({"preprocessing": "transformer_finetuned", "representation": "labse_finetuned"})
    ) == "labse_finetuned"
