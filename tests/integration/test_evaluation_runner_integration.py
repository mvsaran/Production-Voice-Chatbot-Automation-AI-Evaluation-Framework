"""
Integration tests for EvaluationRunner orchestrating multi-service conversations and evaluation aggregation.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.models.base import RoleEnum
from evaluation.adapters.mock_services import (
    InMemoryConversationAdapter,
    MockLLMServiceAdapter,
    MockSpeechToTextAdapter,
    MockTextToSpeechAdapter,
)
from evaluation.models import EvaluationResult, ExecutionResult
from evaluation.runners.evaluation_runner import EvaluationRunner


class DummyPassingEvaluator:
    """Mock AI quality evaluator plugin that always passes."""
    @property
    def name(self) -> str:
        return "DummyPassingPlugin"

    async def evaluate(self, execution_result: ExecutionResult) -> EvaluationResult:
        return EvaluationResult(
            evaluator_name=self.name,
            overall_score=0.95,
            passed=True,
            metric_results=[],
        )


@pytest.mark.asyncio
@pytest.mark.functional
async def test_evaluation_runner_end_to_end(tmp_path: Path) -> None:
    """Test full scenario execution across STT -> LLM -> TTS -> Evaluator pipeline."""
    stt = MockSpeechToTextAdapter()
    llm = MockLLMServiceAdapter()
    tts = MockTextToSpeechAdapter()
    conv = InMemoryConversationAdapter()

    runner = EvaluationRunner(
        stt_service=stt,
        llm_service=llm,
        tts_service=tts,
        conversation_manager=conv,
    )

    runner.register_evaluator(DummyPassingEvaluator())

    audio_path = tmp_path / "test_user.wav"
    audio_path.touch()

    scenario_data = {
        "conversation_id": "INT_RUN_001",
        "scenario_name": "Integration scenario",
        "description": "Integration test description",
        "conversation": [
            {
                "user": "Book ticket tomorrow",
                "audio_path": str(audio_path),
            }
        ],
        "expected_behavior": "Respond with departure questions",
    }

    report = await runner.execute_scenario(scenario_data)

    assert report.conversation_id == "INT_RUN_001"
    assert report.scenario_name == "Integration scenario"
    assert report.passed is True
    assert report.overall_score == 0.95
    assert len(report.evaluator_results) == 1
    assert report.execution_result.recognized_text == "Book ticket tomorrow"
    assert "destination" in report.execution_result.assistant_response or len(report.execution_result.assistant_response) > 0

    # Verify session history was stored
    history = await conv.get_history("INT_RUN_001")
    assert len(history) == 2  # 1 user turn, 1 assistant turn
    assert history[0].role == RoleEnum.USER
    assert history[1].role == RoleEnum.ASSISTANT
