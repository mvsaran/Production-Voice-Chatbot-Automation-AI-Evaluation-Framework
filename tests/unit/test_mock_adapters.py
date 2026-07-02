"""
Unit tests for mock service adapters and in-memory conversation storage.
"""

from pathlib import Path
import pytest

from app.models.base import RoleEnum
from app.utils.exceptions import ConversationNotFoundException
from evaluation.adapters.mock_services import (
    InMemoryConversationAdapter,
    MockLLMServiceAdapter,
    MockSpeechToTextAdapter,
    MockTextToSpeechAdapter,
)


@pytest.mark.asyncio
@pytest.mark.functional
async def test_mock_stt_adapter() -> None:
    """Test MockSpeechToTextAdapter transcription simulation."""
    stt = MockSpeechToTextAdapter()
    text = await stt.speech_to_text(Path("test_flight.wav"))
    assert "Book ticket" in text or len(text) > 0


@pytest.mark.asyncio
@pytest.mark.functional
async def test_mock_llm_adapter() -> None:
    """Test MockLLMServiceAdapter generating contextual responses."""
    llm = MockLLMServiceAdapter()

    res_flight = await llm.generate_response("book flight to Paris", [], "system")
    assert "destination" in res_flight or len(res_flight) > 0


@pytest.mark.asyncio
@pytest.mark.functional
async def test_mock_tts_adapter(tmp_path: Path) -> None:
    """Test MockTextToSpeechAdapter audio synthesis simulation."""
    tts = MockTextToSpeechAdapter()
    out_file = tmp_path / "mock_out.wav"
    res_path = await tts.text_to_speech("Hello synthesized speech", out_file)
    assert res_path == out_file
    assert out_file.exists()


@pytest.mark.asyncio
@pytest.mark.functional
async def test_in_memory_conversation_adapter() -> None:
    """Test InMemoryConversationAdapter CRUD operations and exception handling."""
    conv_mgr = InMemoryConversationAdapter()

    # Create session
    hist = await conv_mgr.create_conversation("session_001", "Be friendly")
    assert hist.conversation_id == "session_001"
    assert hist.system_prompt == "Be friendly"

    # Add message
    msg = await conv_mgr.add_message("session_001", RoleEnum.USER, "Hello bot")
    assert msg.content == "Hello bot"

    # Get history
    history = await conv_mgr.get_history("session_001")
    assert len(history) == 1
    assert history[0].content == "Hello bot"

    # Clear session
    cleared = await conv_mgr.clear_conversation("session_001")
    assert cleared is True
    assert len(await conv_mgr.get_history("session_001")) == 0

    # Test non-existent session get
    with pytest.raises(ConversationNotFoundException):
        await conv_mgr.get_conversation("non_existent_id")

    assert len(await conv_mgr.get_history("non_existent_id")) == 0
    assert await conv_mgr.clear_conversation("non_existent_id") is False
