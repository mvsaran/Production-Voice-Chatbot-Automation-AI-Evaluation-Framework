"""
Abstract Base Classes for Speech Recording, Speech-to-Text (STT), and Text-to-Speech (TTS).

Follows Dependency Inversion Principle (SOLID) to enable seamless provider swapping
(e.g., switching from OpenAI Whisper to Azure Speech, or OpenAI TTS to ElevenLabs).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from app.models.base import AudioMetadata


class BaseSpeechRecorder(ABC):
    """
    Abstract interface for recording microphone audio.
    """

    @abstractmethod
    def start_recording(self) -> None:
        """
        Start capturing audio from the default input microphone stream.
        """
        pass

    @abstractmethod
    def stop_recording(self) -> None:
        """
        Stop capturing audio and buffer the recorded frames in memory.
        """
        pass

    @abstractmethod
    def save_audio(self, filename: Optional[str] = None) -> AudioMetadata:
        """
        Save the buffered recorded audio frames to `/audio/input/` in WAV format.

        Args:
            filename: Optional custom filename. If None, generate a timestamped UUID filename.

        Returns:
            AudioMetadata containing file path, format, duration, and size.
        """
        pass


class BaseSpeechToTextService(ABC):
    """
    Abstract interface for Speech-To-Text (STT) transcription services.
    
    Implementations: OpenAIWhisperService, AzureSpeechToTextService (future).
    """

    @abstractmethod
    async def speech_to_text(self, audio_path: Path | str) -> str:
        """
        Transcribe an audio file into recognized text.

        Args:
            audio_path: Path to the input audio file (WAV format).

        Returns:
            The recognized text transcription as a string.

        Raises:
            EmptyAudioException: If audio file is silent or invalid.
            STTException: If transcription API call fails.
        """
        pass


class BaseTextToSpeechService(ABC):
    """
    Abstract interface for Text-To-Speech (TTS) audio synthesis services.
    
    Implementations: OpenAITTSService, ElevenLabsTTSService (future).
    """

    @abstractmethod
    async def text_to_speech(self, text: str, output_path: Optional[Path | str] = None) -> Path:
        """
        Convert text response into spoken audio and store in `/audio/output/`.

        Args:
            text: The text string to synthesize into speech.
            output_path: Optional destination file path. If None, generate timestamped filename.

        Returns:
            Path object pointing to the generated output audio file.

        Raises:
            TTSException: If speech synthesis API call fails.
        """
        pass
