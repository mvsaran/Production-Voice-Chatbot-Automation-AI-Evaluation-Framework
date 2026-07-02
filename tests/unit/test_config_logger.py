"""
Unit tests for configuration settings validation and loguru logging setup.
"""

from pathlib import Path
import pytest
from pydantic import SecretStr

from app.config.settings import Settings, settings
from app.utils.logger import logger, setup_logger
from evaluation.config import EvaluationSettings


@pytest.mark.smoke
@pytest.mark.functional
def test_app_settings_initialization() -> None:
    """Test default settings initialization and field properties."""
    s = Settings()
    assert s.MODEL_NAME == "gpt-4o"
    assert s.WHISPER_MODEL == "whisper-1"
    assert isinstance(s.OPENAI_API_KEY, SecretStr)


@pytest.mark.functional
def test_app_settings_directory_creation(tmp_path: Path) -> None:
    """Test custom directory paths are created during settings initialization."""
    log_file = tmp_path / "custom_logs" / "app.log"
    audio_dir = tmp_path / "custom_audio"
    s = Settings(LOG_FILE_PATH=log_file, AUDIO_INPUT_DIR=audio_dir)
    assert log_file.parent.exists()
    assert audio_dir.exists()


@pytest.mark.functional
def test_evaluation_settings_initialization() -> None:
    """Test EvaluationSettings default values and singleton instance."""
    es = EvaluationSettings()
    assert es.EVAL_MODEL_NAME == "gpt-4o"
    assert es.TEST_DATA_DIR == Path("test_data")
    assert es.FAIL_ON_REGRESSION is True


@pytest.mark.functional
def test_get_openai_api_key_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test helper function get_openai_api_key_str retrieving secret string."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr("sk-test-secret-key-12345"))
    es = EvaluationSettings()
    key_str = es.get_openai_api_key_str()
    assert key_str == "sk-test-secret-key-12345"


@pytest.mark.functional
def test_ensure_eval_directories_exist(tmp_path: Path) -> None:
    """Test validator creation of evaluation scenario and report directories."""
    test_dir = tmp_path / "test_data_dir"
    rep_html = tmp_path / "reports_html"
    rep_json = tmp_path / "reports_json"
    es = EvaluationSettings(
        TEST_DATA_DIR=test_dir,
        REPORTS_HTML_DIR=rep_html,
        REPORTS_JSON_DIR=rep_json,
    )
    assert test_dir.exists()
    assert rep_html.exists()
    assert rep_json.exists()


@pytest.mark.functional
def test_logger_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test custom loguru logger configuration and directory initialization."""
    log_file = tmp_path / "logs" / "test_app.log"
    monkeypatch.setattr(settings, "LOG_FILE_PATH", log_file)
    monkeypatch.setattr(settings, "LOG_LEVEL", "DEBUG")
    setup_logger()
    assert log_file.parent.exists()
    logger.debug("Test log entry for unit test verification.")
