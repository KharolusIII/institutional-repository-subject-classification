from ir_subject_classification.reporting import create_run_directory
from ir_subject_classification.logging_utils import configure_run_logging


def test_run_manifest_directory_is_never_overwritten(tmp_path):
    first_id, first = create_run_directory(tmp_path, "test")
    second_id, second = create_run_directory(tmp_path, "test")
    assert first.exists() and second.exists()
    assert first_id != second_id


def test_incomplete_run_can_be_resumed(tmp_path):
    first_id, first = create_run_directory(tmp_path, "test", resume=True)
    assert (first / "RUN_INCOMPLETE").exists()
    resumed_id, resumed = create_run_directory(tmp_path, "test", resume=True)
    assert resumed_id == first_id
    assert resumed == first


def test_run_log_persists_messages_and_warnings(tmp_path):
    logger = configure_run_logging(tmp_path)
    logger.info("test message")
    for handler in logger.handlers:
        handler.flush()
    assert "test message" in (tmp_path / "run.log").read_text(encoding="utf-8")
