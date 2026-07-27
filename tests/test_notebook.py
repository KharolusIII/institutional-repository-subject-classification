import nbformat


def test_colab_uses_repo_specific_secret_without_tokenized_remote():
    notebook = nbformat.read("notebooks/01_sedici_subject_classification_colab.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "GITHUB_TOKEN_IR_SUBJECT_CLASSIFICATION" in source
    assert "userdata.get(GITHUB_TOKEN_SECRET_NAME)" in source
    assert "GIT_ASKPASS" in source
    assert "github_token = None" in source
    assert "https://x-access-token:" not in source
    assert "github_token + '@github.com'" not in source
