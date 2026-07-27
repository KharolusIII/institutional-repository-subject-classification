import pandas as pd

from ir_subject_classification.mapping import map_files_to_handles, mapping_coverage


def test_mapping_internal_id_to_handle():
    files = pd.DataFrame({"file_id": ["1", "2"], "txt_path": ["a", "b"]})
    mapping = pd.DataFrame({"internal_id": ["1"], "handle": ["10915/10"]})
    result = map_files_to_handles(files, mapping)
    assert result.loc[0, "handle"] == "10915/10"
    assert pd.isna(result.loc[1, "handle"])
    assert mapping_coverage(result).iloc[0]["mapped_txt_files"] == 1

