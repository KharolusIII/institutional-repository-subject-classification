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
    assert "importlib.invalidate_caches()" in source
    assert "import ir_subject_classification" in source
    assert "auth.authenticate_user()" in source
    assert "PROJECT_DIR = Path('/content/institutional-repository-subject-classification')" in source
    assert "DRIVE_WORKSPACE = Path(BASE_DIR)" in source
    assert "drive_txt_index.csv" in source
    assert "txt_probe = next(fulltext_path.glob" not in source
    assert "https://x-access-token:" not in source
    assert "github_token + '@github.com'" not in source
