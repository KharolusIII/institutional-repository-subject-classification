from ir_subject_classification.config import load_config


def test_execution_scale_profiles():
    smoke = load_config("configs/run_smoke_2026.yaml")
    full_smoke = load_config("configs/run_full_smoke_1k_2026.yaml")
    full_20k = load_config("configs/run_full_20k_2026.yaml")
    full_20k_v2 = load_config("configs/run_full_20k_v2_2026.yaml")
    full_20k_v3 = load_config("configs/run_full_20k_v3_2026.yaml")
    protocol_smoke_v3 = load_config("configs/run_protocol_smoke_1k_v3_2026.yaml")
    full_corpus_v3 = load_config("configs/run_full_corpus_v3_2026.yaml")
    full_final = load_config("configs/run_full_final_2026.yaml")
    medium = load_config("configs/run_20k_2026.yaml")
    full = load_config("configs/run_full_corpus_2026.yaml")
    assert smoke["sampling"]["target_n"] == 1000
    assert full_smoke["sampling"]["target_n"] == 1000
    assert full_smoke["representations"]["enabled"] == ["bow", "tfidf", "bm25", "sbert", "labse"]
    assert full_smoke["representations"]["embeddings"]["modes"] == [
        "chunked_length_weighted_mean"
    ]
    assert full_20k["sampling"]["target_n"] == 20000
    assert full_20k["representations"]["enabled"] == [
        "bow",
        "tfidf",
        "bm25",
        "sbert",
        "labse",
    ]
    assert full_20k["classifiers"]["enabled"] == ["logreg", "linear_svc", "sgd"]
    assert full_20k["experiment"]["name"] == "full_20k_231_combinations_2026"
    assert full_20k_v2["language"]["backend"] == "langid"
    assert full_20k_v2["split"]["calibration_size"] == 0.10
    assert full_20k_v2["thresholds"]["mode"] == "per_label_threshold"
    assert full_20k_v2["finetuning"]["enabled"] is True
    assert set(full_20k_v2["finetuning"]["models"]) == {"sbert_finetuned", "labse_finetuned"}
    assert full_20k_v3["data"]["preserve_text_segments"] is True
    assert full_20k_v3["data"]["max_fulltext_chars"] == 400000
    assert full_20k_v3["features"]["field_weights"]["fulltext"] == 0.60
    assert full_20k_v3["evaluation"]["secondary_test_families"] == [
        "bow", "tfidf", "bm25", "sbert_frozen", "labse_frozen",
        "sbert_finetuned", "labse_finetuned",
    ]
    assert protocol_smoke_v3["sampling"]["target_n"] == 1000
    assert protocol_smoke_v3["finetuning"]["max_epochs"] == 1
    assert protocol_smoke_v3["finetuning"]["early_stopping"]["enabled"] is False
    assert full_20k_v3["finetuning"]["max_epochs"] == 5
    assert full_20k_v3["finetuning"]["early_stopping"]["enabled"] is True
    assert full_corpus_v3["sampling"]["target_n"] is None
    assert full_final["sampling"]["target_n"] is None
    assert full_final["labels"]["top_k"] == 37
    assert full_final["representations"]["enabled"] == [
        "bow",
        "tfidf",
        "bm25",
        "sbert",
        "labse",
    ]
    assert full_final["classifiers"]["enabled"] == ["logreg", "linear_svc", "sgd"]
    assert full_final["experiment"]["stage"] == "all"
    sparse = {"bow", "tfidf", "bm25"}
    dense = {"sbert", "labse"}
    representations = set(full_final["representations"]["enabled"])
    combinations = (
        len(full_final["preprocessing"]["modes"])
        * len(full_final["features"]["sets"])
        * len(representations & sparse)
        * len(full_final["classifiers"]["enabled"])
        + len(full_final["features"]["sets"])
        * len(representations & dense)
        * len(full_final["representations"]["embeddings"]["modes"])
        * len(full_final["classifiers"]["enabled"])
    )
    assert combinations == 231
    assert full_final["artifacts"]["include_text_in_dataset_splits"] is False
    assert medium["sampling"]["target_n"] == 20000
    assert full["sampling"]["target_n"] is None
    assert full["experiment"]["resume"] is True
