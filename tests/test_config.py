from ir_subject_classification.config import load_config


def test_execution_scale_profiles():
    smoke = load_config("configs/run_smoke_2026.yaml")
    medium = load_config("configs/run_20k_2026.yaml")
    full = load_config("configs/run_full_corpus_2026.yaml")
    assert smoke["sampling"]["target_n"] == 1000
    assert medium["sampling"]["target_n"] == 20000
    assert full["sampling"]["target_n"] is None
