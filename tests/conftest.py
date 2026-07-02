"""
Pytest fixtures and configuration for the AI Evaluation Framework.

Provides dependency-injected mock voice chatbot services, scenario paths,
and ready-to-run EvaluationRunner and DeepEvalRunner fixtures for test suites.
"""

from pathlib import Path
from typing import Any, Dict
import pytest
from evaluation.deepeval.deepeval_runner import DeepEvalRunner
from evaluation.runner import (
    EvaluationRunner,
    InMemoryConversationAdapter,
    MockLLMServiceAdapter,
    MockSpeechToTextAdapter,
    MockTextToSpeechAdapter,
)


@pytest.fixture
def mock_stt_service() -> MockSpeechToTextAdapter:
    """Fixture providing a mock Speech-to-Text adapter."""
    return MockSpeechToTextAdapter()


@pytest.fixture
def mock_llm_service() -> MockLLMServiceAdapter:
    """Fixture providing a mock LLM service adapter."""
    return MockLLMServiceAdapter()


@pytest.fixture
def mock_tts_service() -> MockTextToSpeechAdapter:
    """Fixture providing a mock Text-to-Speech adapter."""
    return MockTextToSpeechAdapter()


@pytest.fixture
def mock_conv_manager() -> InMemoryConversationAdapter:
    """Fixture providing an in-memory conversation session manager."""
    return InMemoryConversationAdapter()


@pytest.fixture
def eval_runner(
    mock_stt_service: MockSpeechToTextAdapter,
    mock_llm_service: MockLLMServiceAdapter,
    mock_tts_service: MockTextToSpeechAdapter,
    mock_conv_manager: InMemoryConversationAdapter,
) -> EvaluationRunner:
    """Fixture providing a configured EvaluationRunner with dependency-injected mock services."""
    return EvaluationRunner(
        stt_service=mock_stt_service,
        llm_service=mock_llm_service,
        tts_service=mock_tts_service,
        conversation_manager=mock_conv_manager,
    )


@pytest.fixture
def deepeval_runner() -> DeepEvalRunner:
    """Fixture providing a DeepEvalRunner configured in simulation mode for offline CI reliability."""
    return DeepEvalRunner(simulation_mode=True)


@pytest.fixture
def sample_json_scenario_path() -> Path:
    """Fixture providing path to sample JSON test scenario."""
    return Path("test_data/conversations/sample_flight_booking.json")


@pytest.fixture
def sample_yaml_scenario_path() -> Path:
    """Fixture providing path to sample YAML test scenario."""
    return Path("test_data/conversations/sample_flight_booking.yaml")
