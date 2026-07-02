"""
Backward-compatibility re-export shim for EvaluationRunner and mock adapters.
Please import from `evaluation.runners.evaluation_runner` and `evaluation.adapters.mock_services` in new code.
"""

from evaluation.adapters.mock_services import (
    InMemoryConversationAdapter,
    MockLLMServiceAdapter,
    MockSpeechToTextAdapter,
    MockTextToSpeechAdapter,
)
from evaluation.runners.evaluation_runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
    "InMemoryConversationAdapter",
    "MockLLMServiceAdapter",
    "MockSpeechToTextAdapter",
    "MockTextToSpeechAdapter",
]
