"""
Custom exception hierarchy for the Voice Chatbot Automation Framework.

Provides specialized exceptions for every failure domain:
- Audio recording / Empty audio
- Network failures and timeouts
- Invalid API keys and authentication errors
- Speech recognition (STT) failures
- LLM generation failures
- Text-To-Speech (TTS) synthesis failures
"""

from typing import Optional


class VoiceAutomationException(Exception):
    """Base exception for all errors within the Voice Automation Framework."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationException(VoiceAutomationException):
    """Raised when environment variables or configuration settings are invalid or missing."""
    pass


class AuthenticationException(VoiceAutomationException):
    """Raised when API keys (OpenAI or Azure) are invalid, expired, or unauthorized."""
    pass


class NetworkTimeoutException(VoiceAutomationException):
    """Raised when external API calls experience network timeouts or connection failures."""
    pass


class AudioException(VoiceAutomationException):
    """Base exception for audio hardware or file processing errors."""
    pass


class EmptyAudioException(AudioException):
    """Raised when recorded or uploaded audio is empty, silent, or below minimum duration thresholds."""
    pass


class STTException(VoiceAutomationException):
    """Raised when Speech-To-Text (e.g., Whisper API) transcription fails."""
    pass


class LLMException(VoiceAutomationException):
    """Raised when Language Model (OpenAI GPT-4.1 / GPT-4o / Azure) response generation fails."""
    pass


class TTSException(VoiceAutomationException):
    """Raised when Text-To-Speech (OpenAI TTS / ElevenLabs) synthesis fails."""
    pass


class ConversationNotFoundException(VoiceAutomationException):
    """Raised when requested conversation ID does not exist in memory or storage."""
    pass


class EvaluationException(VoiceAutomationException):
    """Base exception for all AI Quality Assurance and Evaluation failures."""
    pass


class ScenarioLoadException(EvaluationException):
    """Raised when test scenario files (.json / .yaml) cannot be read or parsed."""
    pass


class ScenarioValidationException(EvaluationException):
    """Raised when scenario definition syntax or required schema fields are missing."""
    pass


class EvaluatorExecutionException(EvaluationException):
    """Raised when an evaluation engine (DeepEval, RAGAS, Voice QA) fails during execution."""
    pass


class MetricCalculationException(EvaluationException):
    """Raised when individual metric calculations error out or breach mathematical limits."""
    pass


class AggregationException(EvaluationException):
    """Raised when multi-evaluator result merging or report compilation fails."""
    pass

