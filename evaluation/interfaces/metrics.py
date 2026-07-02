"""
Abstract Metric, Voice, Conversation, and Security Evaluator Interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from evaluation.models import (
    ConversationValidationResult,
    MetricResult,
    SecurityEvaluationResult,
    TurnExecutionArtifacts,
    VoiceQualityResult,
)


class BaseMetricEvaluator(ABC):
    """Abstract interface for individual AI response evaluation metrics."""

    @abstractmethod
    async def evaluate(
        self,
        prompt: str,
        response: str,
        expected_response: Optional[str] = None,
        context: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Evaluate an AI response against a prompt and reference criteria."""
        pass


class BaseVoiceEvaluator(ABC):
    """Abstract interface for evaluating speech recognition quality and pipeline latencies."""

    @abstractmethod
    async def evaluate_voice(
        self,
        expected_text: str,
        recognized_text: str,
        stt_latency_sec: float,
        tts_latency_sec: float,
        total_latency_sec: float,
    ) -> VoiceQualityResult:
        """Evaluate Word Error Rate (WER), Character Error Rate (CER), and SLA adherence."""
        pass


class BaseConversationValidator(ABC):
    """Abstract interface for multi-turn conversation flow validation."""

    @abstractmethod
    async def validate_conversation(
        self,
        conversation_id: str,
        turns: List[TurnExecutionArtifacts],
    ) -> ConversationValidationResult:
        """Validate context preservation, loop detection, and task completion across turns."""
        pass


class BaseSecurityEvaluator(ABC):
    """Abstract interface for prompt injection protection, toxicity, and PII leakage testing."""

    @abstractmethod
    async def evaluate_security(
        self,
        prompt: str,
        response: str,
    ) -> SecurityEvaluationResult:
        """Analyze prompt/response pairs for security vulnerabilities and safety compliance."""
        pass
