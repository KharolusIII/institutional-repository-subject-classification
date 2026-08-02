import nbformat


def test_colab_uses_repo_specific_secret_without_tokenized_remote():
    notebook = nbformat.read("notebooks/01_sedici_subject_classification_colab.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "GITHUB_TOKEN_IR_SUBJECT_CLASSIFICATION" in source
    assert "userdata.get(GITHUB_TOKEN_SECRET_NAME)" in source
    assert "GIT_ASKPASS" in source
    assert "GIT_ASKPASS_REQUIRE" in source
    assert "git', 'ls-remote" in source
    assert "capture_output=True" in source
    assert "replace(github_token, '[REDACTED]')" in source
    assert "github_token = None" in source
    assert "sys.path.insert(0, source_dir)" in source
    assert "del sys.modules[module_name]" in source
    assert "importlib.invalidate_caches()" in source
    assert "import ir_subject_classification" in source
    assert "auth.authenticate_user()" in source
    assert "PROJECT_DIR = Path('/content/institutional-repository-subject-classification')" in source
    assert "DRIVE_WORKSPACE = Path(BASE_DIR)" in source
    assert "drive_txt_index.csv" in source
    assert "drive_text_cache" in source
    assert "'full-smoke': 'configs/run_full_smoke_1k_2026.yaml'" in source
    assert "'full-final': 'configs/run_full_final_2026.yaml'" in source
    assert "'full-20k': 'configs/run_full_20k_2026.yaml'" in source
    assert "EXECUTION_PROFILE = 'full-20k-v3'" in source
    assert "'full-20k-v3': 'configs/run_full_20k_v3_2026.yaml'" in source
    assert "'full-v3': 'configs/run_full_corpus_v3_2026.yaml'" in source
    assert "'full-20k-v2': 'configs/run_full_20k_v2_2026.yaml'" in source
    assert "RUN_FINE_TUNING = True" in source
    assert "results_test_comparative.csv" in source
    assert "paired_bootstrap_vs_global.csv" in source
    assert "FINETUNING_DIR = Path(RUN_DIR) / 'finetuning'" in source
    assert "dataset_prepared_langid_v2.parquet" in source
    assert "dataset_prepared_segmented_v3.parquet" in source
    assert "fulltext_corpus_segmented_v3.parquet" in source
    assert "EXECUTION_STAGE = 'all'" in source
    assert "USE_DUMMY_DATA = False" in source
    assert "materialized_fulltext_parquet" in source
    assert "materialized_dataset_parquet" in source
    assert "embeddings']['cache_dir" in source
    assert "config['experiment']['resume'] = RESUME_CACHES" in source
    assert "txt_probe = next(fulltext_path.glob" not in source
    assert "https://x-access-token:" not in source
    assert "github_token + '@github.com'" not in source
