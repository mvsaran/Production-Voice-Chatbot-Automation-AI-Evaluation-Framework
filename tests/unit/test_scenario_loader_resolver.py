"""
Unit tests for ScenarioLoader parsing/validation and ServiceResolver dependency injection.
"""

from pathlib import Path
import pytest

from evaluation.adapters.mock_services import (
    InMemoryConversationAdapter,
    MockLLMServiceAdapter,
    MockSpeechToTextAdapter,
    MockTextToSpeechAdapter,
    ServiceResolver,
)
from evaluation.runners.scenario_loader import ScenarioLoader


@pytest.fixture
def loader() -> ScenarioLoader:
    """Fixture providing a clean ScenarioLoader instance."""
    return ScenarioLoader()


@pytest.mark.smoke
@pytest.mark.functional
def test_scenario_loader_valid_json(loader: ScenarioLoader, tmp_path: Path) -> None:
    """Test loading a well-formed JSON scenario file."""
    f = tmp_path / "valid.json"
    f.write_text(
        '{"conversation_id": "TEST_001", "description": "desc", "conversation": [{"role": "user", "content": "hi"}]}',
        encoding="utf-8",
    )
    res = loader.load_scenario(f)
    assert res["conversation_id"] == "TEST_001"
    assert len(res["conversation"]) == 1


@pytest.mark.functional
def test_scenario_loader_valid_yaml(loader: ScenarioLoader, tmp_path: Path) -> None:
    """Test loading a well-formed YAML scenario file."""
    f = tmp_path / "valid.yaml"
    f.write_text(
        "conversation_id: TEST_002\ndescription: yaml desc\nconversation:\n  - role: user\n    content: hello\n",
        encoding="utf-8",
    )
    res = loader.load_scenario(f)
    assert res["conversation_id"] == "TEST_002"
    assert res["description"] == "yaml desc"


@pytest.mark.functional
@pytest.mark.negative
def test_scenario_loader_missing_file_raises(loader: ScenarioLoader) -> None:
    """Test that missing file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        loader.load_scenario(Path("non_existent_dir/scenario.json"))


@pytest.mark.functional
@pytest.mark.negative
def test_scenario_loader_malformed_json_raises(loader: ScenarioLoader, tmp_path: Path) -> None:
    """Test that malformed JSON raises ValueError with syntax details."""
    f = tmp_path / "bad.json"
    f.write_text('{"conversation_id": "TEST_003", syntax_error...}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        loader.load_scenario(f)


@pytest.mark.functional
@pytest.mark.negative
def test_scenario_loader_malformed_yaml_raises(loader: ScenarioLoader, tmp_path: Path) -> None:
    """Test that malformed YAML raises ValueError."""
    f = tmp_path / "bad.yaml"
    f.write_text("conversation_id: [unclosed list\n  - foo: : bar", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        loader.load_scenario(f)


@pytest.mark.functional
@pytest.mark.negative
def test_scenario_loader_missing_required_keys_raises(loader: ScenarioLoader, tmp_path: Path) -> None:
    """Test that JSON missing required schema fields raises ValueError."""
    f = tmp_path / "missing_keys.json"
    f.write_text('{"description": "missing cid and conv"}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        loader.load_scenario(f)


@pytest.mark.functional
@pytest.mark.negative
def test_scenario_loader_non_list_conversation_raises(loader: ScenarioLoader, tmp_path: Path) -> None:
    """Test that scenario where conversation is not a list raises ValueError."""
    f = tmp_path / "bad_conv.json"
    f.write_text('{"conversation_id": "C1", "description": "desc", "conversation": "not a list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        loader.load_scenario(f)


@pytest.mark.functional
def test_service_resolver_defaults() -> None:
    """Test ServiceResolver defaulting to simulation mock adapters when None provided."""
    assert isinstance(ServiceResolver.resolve_stt(), MockSpeechToTextAdapter)
    assert isinstance(ServiceResolver.resolve_llm(), MockLLMServiceAdapter)
    assert isinstance(ServiceResolver.resolve_tts(), MockTextToSpeechAdapter)
    assert isinstance(ServiceResolver.resolve_conversation_manager(), InMemoryConversationAdapter)


@pytest.mark.functional
def test_service_resolver_custom_injection() -> None:
    """Test ServiceResolver using custom injected service instances."""
    custom_stt = MockSpeechToTextAdapter()
    custom_llm = MockLLMServiceAdapter()
    custom_tts = MockTextToSpeechAdapter()
    custom_conv = InMemoryConversationAdapter()

    assert ServiceResolver.resolve_stt(custom_stt) is custom_stt
    assert ServiceResolver.resolve_llm(custom_llm) is custom_llm
    assert ServiceResolver.resolve_tts(custom_tts) is custom_tts
    assert ServiceResolver.resolve_conversation_manager(custom_conv) is custom_conv
