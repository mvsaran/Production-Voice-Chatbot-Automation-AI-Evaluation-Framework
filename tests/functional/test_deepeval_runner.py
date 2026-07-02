"""
Functional tests for the DeepEvalRunner evaluator integration.

Verifies that DeepEval quality metrics (Answer Relevancy, Faithfulness, Hallucination,
and Answer Correctness) execute, score correctly, and handle timeouts gracefully.
"""

import pytest
from datetime import datetime, timezone
from evaluation.deepeval.deepeval_runner import DeepEvalRunner
from evaluation.models import ExecutionResult


@pytest.fixture
def sample_execution_result() -> ExecutionResult:
    """Fixture providing a realistic ExecutionResult for evaluating DeepEval metrics."""
    return ExecutionResult(
        conversation_id="TEST_DEEPEVAL_001",
        scenario_name="Flight Booking Scenario",
        recognized_text="I want to book a flight to Paris for tomorrow morning.",
        assistant_response="I can help you book a flight to Paris for tomorrow morning. What time would you prefer to depart?",
        conversation_history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "I want to book a flight to Paris for tomorrow morning."},
            {"role": "assistant", "content": "I can help you book a flight to Paris for tomorrow morning. What time would you prefer to depart?"}
        ],
        audio_input_path="audio/input/test.wav",
        audio_output_path="audio/output/test.wav",
        speech_latency=0.12,
        llm_latency=0.45,
        tts_latency=0.20,
        total_latency=0.77,
        timestamp=datetime.now(timezone.utc),
        metadata={
            "expected_behavior": "Assistant confirms Paris destination and asks for departure time.",
            "context": ["User requested flight booking to Paris tomorrow morning."]
        }
    )


@pytest.mark.smoke
@pytest.mark.functional
@pytest.mark.asyncio
async def test_deepeval_runner_execution_simulation(
    deepeval_runner: DeepEvalRunner,
    sample_execution_result: ExecutionResult,
) -> None:
    """Test executing all 4 DeepEval metrics in simulation mode."""
    assert deepeval_runner.name == "DeepEval"

    res = await deepeval_runner.evaluate(sample_execution_result)

    assert res.evaluator_name == "DeepEval"
    assert res.passed is True
    assert res.overall_score > 0.8
    assert len(res.metric_results) == 4

    metric_names = [m.metric_name for m in res.metric_results]
    assert "Answer Relevancy" in metric_names
    assert "Faithfulness" in metric_names
    assert "Hallucination" in metric_names
    assert "Answer Correctness" in metric_names

    for m in res.metric_results:
        assert m.passed is True
        assert m.error_message is None
        assert m.execution_time_ms > 0.0


@pytest.mark.functional
@pytest.mark.asyncio
async def test_deepeval_runner_with_timeout_fallback(sample_execution_result: ExecutionResult) -> None:
    """Test that DeepEvalRunner handles simulated timeouts without crashing."""
    # Initialize runner with very short timeout to test resilience
    runner = DeepEvalRunner(timeout_seconds=0.0001, simulation_mode=False)

    res = await runner.evaluate(sample_execution_result)
    assert res.evaluator_name == "DeepEval"
    # Even if metrics time out or fallback due to missing SDK/key, it returns a structured EvaluationResult
    assert isinstance(res.overall_score, float)
