from ir_subject_classification.ingestion import allocate_text_budget


def test_text_budget_is_shared_across_files_instead_of_first_file_truncation():
    retained = allocate_text_budget(["a" * 100, "b" * 100], 80)
    assert list(map(len, retained)) == [40, 40]
    assert retained[0].startswith("a") and retained[1].startswith("b")

