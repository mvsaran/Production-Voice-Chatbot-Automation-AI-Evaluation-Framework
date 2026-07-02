"""
Integration tests specifically targeting OpenAITTSService edge cases and directory handling.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.config.settings import settings
from app.services.speech_service import OpenAITTSService
from app.utils.exceptions import TTSException


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Fixture providing a mock AsyncOpenAI client with speech synthesis endpoint."""
    client = MagicMock()
    client.audio.speech.create = AsyncMock()
    return client


@pytest.mark.asyncio
@pytest.mark.functional
async def test_tts_creates_parent_directory(mock_openai_client: MagicMock, tmp_path: Path) -> None:
    """Test that OpenAITTSService automatically creates parent directory structure if missing."""
    mock_response = MagicMock()
    mock_response.aread = AsyncMock(return_value=b"RIFF synthesized audio bytes")
    mock_openai_client.audio.speech.create.return_value = mock_response

    service = OpenAITTSService(client=mock_openai_client)
    nested_dir = tmp_path / "deep" / "nested" / "audio"
    out_path = nested_dir / "output.wav"

    assert not nested_dir.exists()
    res = await service.text_to_speech("Testing nested dir creation", out_path)
    assert res == out_path
    assert nested_dir.exists()
    assert out_path.read_bytes() == b"RIFF synthesized audio bytes"


@pytest.mark.asyncio
@pytest.mark.functional
async def test_tts_voice_and_model_settings(mock_openai_client: MagicMock, tmp_path: Path) -> None:
    """Test that OpenAITTSService passes correct model and voice token settings to API."""
    mock_response = MagicMock()
    mock_response.aread = AsyncMock(return_value=b"RIFF audio")
    mock_openai_client.audio.speech.create.return_value = mock_response

    service = OpenAITTSService(client=mock_openai_client)
    out_path = tmp_path / "test.wav"

    await service.text_to_speech("Verify voice configuration", out_path)
    mock_openai_client.audio.speech.create.assert_called_once_with(
        model=settings.TTS_MODEL,
        voice=settings.TTS_VOICE,
        input="Verify voice configuration",
    )


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.negative
async def test_tts_whitespace_only_text_raises(mock_openai_client: MagicMock, tmp_path: Path) -> None:
    """Test that passing whitespace-only string to TTS raises TTSException."""
    service = OpenAITTSService(client=mock_openai_client)
    with pytest.raises(TTSException, match="empty text string"):
        await service.text_to_speech("   \n\t  ", tmp_path / "empty.wav")
