from ir_subject_classification.config import load_config


def test_execution_scale_profiles():
    smoke = load_config("configs/run_smoke_2026.yaml")
    full_smoke = load_config("configs/run_full_smoke_1k_2026.yaml")
    full_final = load_config("configs/run_full_final_2026.yaml")
    medium = load_config("configs/run_20k_2026.yaml")
    full = load_config("configs/run_full_corpus_2026.yaml")
    assert smoke["sampling"]["target_n"] == 1000
    assert full_smoke["sampling"]["target_n"] == 1000
    assert full_smoke["representations"]["enabled"] == ["bow", "tfidf", "bm25", "sbert", "labse"]
    assert full_smoke["representations"]["embeddings"]["modes"] == [
        "chunked_length_weighted_mean"
    ]
    assert full_final["sampling"]["target_n"] is None
    assert full_final["labels"]["top_k"] == 37
    assert full_final["representations"]["enabled"] == ["bm25"]
    assert full_final["classifiers"]["enabled"] == ["sgd"]
    assert full_final["artifacts"]["include_text_in_dataset_splits"] is False
    assert medium["sampling"]["target_n"] == 20000
    assert full["sampling"]["target_n"] is None
    assert full["experiment"]["resume"] is True
