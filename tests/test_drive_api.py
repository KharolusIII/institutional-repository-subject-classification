from pathlib import Path

import pytest

from ir_subject_classification.drive_api import normalize_my_drive_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "/content/drive/MyDrive/A___Maestria_en_ID/Datos_SEDICI/SEDICI_FullText_TXT",
            ["A___Maestria_en_ID", "Datos_SEDICI", "SEDICI_FullText_TXT"],
        ),
        (
            r"\content\drive\My Drive\course\data",
            ["course", "data"],
        ),
    ],
)
def test_normalize_my_drive_path(path: str, expected: list[str]) -> None:
    assert normalize_my_drive_path(path) == expected


def test_normalize_my_drive_path_rejects_non_drive_path() -> None:
    with pytest.raises(ValueError, match="below MyDrive"):
        normalize_my_drive_path(Path("/tmp/data"))
