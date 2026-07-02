"""
Production Speech Services Implementation using OpenAI Whisper and OpenAI TTS.

Implements BaseSpeechToTextService and BaseTextToSpeechService interfaces with
full async I/O, error handling, and custom exception mapping.
"""

import uuid
from pathlib import Path
from typing import Optional
import aiofiles
from loguru import logger
from openai import AsyncOpenAI

from app.config.settings import settings
from app.speech.base import BaseSpeechToTextService, BaseTextToSpeechService
from app.utils.exceptions import EmptyAudioException, STTException, TTSException


class OpenAIWhisperService(BaseSpeechToTextService):
    """
    Production Speech-to-Text service utilizing OpenAI Whisper API.
    """

    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        """
        Initialize the Whisper STT service.

        Args:
            client: Optional AsyncOpenAI client instance for dependency injection and mocking.
        """
        self._client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())

    async def speech_to_text(self, audio_path: Path | str) -> str:
        """
        Transcribe an audio file into recognized text using Whisper.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            Recognized transcription text.

        Raises:
            EmptyAudioException: If audio file does not exist or size is 0 bytes.
            STTException: If Whisper API transcription fails.
        """
        path = Path(audio_path)
        logger.debug(f"Starting Whisper transcription for audio file: {path}")

        if not path.exists() or path.stat().st_size == 0:
            err_msg = f"Audio file '{path}' does not exist or is empty (0 bytes)."
            logger.error(err_msg)
            raise EmptyAudioException(err_msg)

        try:
            with open(path, "rb") as audio_file:
                response = await self._client.audio.transcriptions.create(
                    model=settings.WHISPER_MODEL,
                    file=audio_file,
                )
            text: str = str(getattr(response, "text", response)).strip()
            logger.info(f"Whisper transcription successful | Length: {len(text)} chars")
            return text
        except Exception as e:
            err_msg = f"Whisper API transcription failed: {str(e)}"
            logger.error(err_msg)
            raise STTException(err_msg) from e


class OpenAITTSService(BaseTextToSpeechService):
    """
    Production Text-to-Speech service utilizing OpenAI Speech Synthesis API.
    """

    def __init__(self, client: Optional[AsyncOpenAI] = None) -> None:
        """
        Initialize the OpenAI TTS service.

        Args:
            client: Optional AsyncOpenAI client instance for dependency injection and mocking.
        """
        self._client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())

    async def text_to_speech(self, text: str, output_path: Optional[Path | str] = None) -> Path:
        """
        Convert text string into spoken WAV audio and save to disk.

        Args:
            text: Text to synthesize into speech.
            output_path: Optional destination path. If None, auto-generates timestamped filename.

        Returns:
            Path object pointing to the created audio file.

        Raises:
            TTSException: If OpenAI TTS API synthesis fails.
        """
        if not text or not text.strip():
            err_msg = "Cannot synthesize speech from empty text string."
            logger.error(err_msg)
            raise TTSException(err_msg)

        out_path = Path(
            output_path or settings.AUDIO_OUTPUT_DIR / f"tts_{uuid.uuid4().hex}.wav"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Starting TTS synthesis for text ({len(text)} chars) -> {out_path}")

        try:
            response = await self._client.audio.speech.create(
                model=settings.TTS_MODEL,
                voice=settings.TTS_VOICE,
                input=text,
            )
            async with aiofiles.open(out_path, "wb") as f:
                content = await response.aread() if hasattr(response, "aread") else response.content
                await f.write(content)

            logger.info(f"TTS synthesis successful | Saved to: {out_path}")
            return out_path
        except Exception as e:
            err_msg = f"OpenAI TTS API speech synthesis failed: {str(e)}"
            logger.error(err_msg)
            raise TTSException(err_msg) from e
