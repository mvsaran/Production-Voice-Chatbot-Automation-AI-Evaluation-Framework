"""
Integration tests for Speech Services (Whisper STT & OpenAI TTS) with mocked external APIs.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.config.settings import settings
from app.services.speech_service import OpenAITTSService, OpenAIWhisperService
from app.utils.exceptions import EmptyAudioException, STTException, TTSException


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Fixture providing a mock AsyncOpenAI client with async transcriptions and speech endpoints."""
    client = MagicMock()
    client.audio.transcriptions.create = AsyncMock()
    client.audio.speech.create = AsyncMock()
    return client


@pytest.mark.asyncio
@pytest.mark.functional
async def test_whisper_service_success(mock_openai_client: MagicMock, tmp_path: Path) -> None:
    """Test OpenAIWhisperService successful transcription with mocked API call."""
    mock_response = MagicMock()
    mock_response.text = "Hello, this is a mocked Whisper transcription."
    mock_openai_client.audio.transcriptions.create.return_value = mock_response

    service = OpenAIWhisperService(client=mock_openai_client)

    audio_file = tmp_path / "test_input.wav"
    audio_file.write_bytes(b"dummy audio data content for test")

    transcription = await service.speech_to_text(audio_file)
    assert transcription == "Hello, this is a mocked Whisper transcription."
    mock_openai_client.audio.transcriptions.create.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.negative
async def test_whisper_service_missing_file(mock_openai_client: MagicMock) -> None:
    """Test that transcribing a non-existent audio file raises EmptyAudioException."""
    service = OpenAIWhisperService(client=mock_openai_client)
    with pytest.raises(EmptyAudioException, match="does not exist"):
        await service.speech_to_text("non_existent_audio.wav")


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.negative
async def test_whisper_service_empty_file(mock_openai_client: MagicMock, tmp_path: Path) -> None:
    """Test that transcribing a 0-byte audio file raises EmptyAudioException."""
    service = OpenAIWhisperService(client=mock_openai_client)
    empty_file = tmp_path / "silent.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(EmptyAudioException, match="is empty"):
        await service.speech_to_text(empty_file)


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.negative
async def test_whisper_service_api_failure(mock_openai_client: MagicMock, tmp_path: Path) -> None:
    """Test that an OpenAI API exception during transcription raises STTException."""
    mock_openai_client.audio.transcriptions.create.side_effect = RuntimeError("OpenAI 500 Internal Server Error")
    service = OpenAIWhisperService(client=mock_openai_client)

    audio_file = tmp_path / "test_fail.wav"
    audio_file.write_bytes(b"dummy audio data content")

    with pytest.raises(STTException, match="Whisper API transcription failed"):
        await service.speech_to_text(audio_file)


@pytest.mark.asyncio
@pytest.mark.functional
async def test_tts_service_success(mock_openai_client: MagicMock, tmp_path: Path) -> None:
    """Test OpenAITTSService successful speech synthesis and WAV file creation."""
    mock_response = MagicMock()
    mock_response.aread = AsyncMock(return_value=b"RIFF dummy wav binary audio headers and data")
    mock_openai_client.audio.speech.create.return_value = mock_response

    service = OpenAITTSService(client=mock_openai_client)
    out_path = tmp_path / "output" / "speech.wav"

    res_path = await service.text_to_speech("Welcome to the chatbot", out_path)
    assert res_path == out_path
    assert out_path.exists()
    assert out_path.read_bytes() == b"RIFF dummy wav binary audio headers and data"
    mock_openai_client.audio.speech.create.assert_called_once_with(
        model=settings.TTS_MODEL,
        voice=settings.TTS_VOICE,
        input="Welcome to the chatbot",
    )


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.negative
async def test_tts_service_empty_text(mock_openai_client: MagicMock) -> None:
    """Test that passing an empty string to TTS raises TTSException."""
    service = OpenAITTSService(client=mock_openai_client)
    with pytest.raises(TTSException, match="empty text string"):
        await service.text_to_speech("")


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.negative
async def test_tts_service_api_failure(mock_openai_client: MagicMock, tmp_path: Path) -> None:
    """Test that an API exception during TTS synthesis raises TTSException."""
    mock_openai_client.audio.speech.create.side_effect = RuntimeError("TTS rate limit exceeded")
    service = OpenAITTSService(client=mock_openai_client)

    with pytest.raises(TTSException, match="OpenAI TTS API speech synthesis failed"):
        await service.text_to_speech("Test speech", tmp_path / "fail.wav")
