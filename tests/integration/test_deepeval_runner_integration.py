"""
Integration tests for DeepEval evaluation engine and metric execution.
"""

from datetime import datetime, timezone
import pytest

from evaluation.engines.deepeval.runner import DeepEvalRunner
from evaluation.models import ExecutionResult


@pytest.fixture
def sample_execution_result() -> ExecutionResult:
    """Fixture providing a complete ExecutionResult payload for metric evaluation."""
    return ExecutionResult(
        conversation_id="DEEP_INT_001",
        scenario_name="Flight Booking Scenario",
        recognized_text="I would like to book a flight from New York to London departing next Friday.",
        assistant_response="Certainly! I can help you book a flight from New York (JFK) to London (LHR) next Friday. What class would you prefer?",
        conversation_history=[
            {"role": "user", "content": "Hello there."},
            {"role": "assistant", "content": "Welcome to AI Airways! How can I assist you?"},
        ],
        audio_input_path="test_in.wav",
        audio_output_path="test_out.wav",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.functional
async def test_deepeval_runner_integration(sample_execution_result: ExecutionResult) -> None:
    """Test DeepEvalRunner executing all 4 core quality metrics in simulation or live mode."""
    runner = DeepEvalRunner(simulation_mode=True)
    assert runner.name == "DeepEval"

    res = await runner.evaluate(sample_execution_result)

    assert res.evaluator_name == "DeepEval"
    assert res.passed is True
    assert res.overall_score >= 0.8
    assert len(res.metric_results) == 4

    metric_names = {m.metric_name for m in res.metric_results}
    expected_metrics = {"Answer Relevancy", "Faithfulness", "Hallucination", "Answer Correctness"}
    assert expected_metrics == metric_names

    for metric in res.metric_results:
        assert metric.passed is True
        assert metric.error_message is None
