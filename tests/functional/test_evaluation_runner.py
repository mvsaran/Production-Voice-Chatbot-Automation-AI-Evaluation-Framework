"""
Functional and smoke tests for the EvaluationRunner execution engine.

Tests scenario parsing (JSON/YAML), pipeline orchestration (Speech -> LLM -> TTS),
evaluator plugin registration, and error resilience when individual stages fail.
"""

import pytest
from pathlib import Path
from typing import Any
from evaluation.deepeval.deepeval_runner import DeepEvalRunner
from evaluation.models import EvaluationResult, ExecutionResult
from evaluation.interfaces.evaluator_interface import Evaluator
from evaluation.runner import EvaluationRunner


class DummyCustomEvaluator(Evaluator):
    """Custom mock evaluator for testing multi-evaluator aggregation."""
    @property
    def name(self) -> str:
        return "CustomEnterpriseMetric"

    async def evaluate(self, execution_result: ExecutionResult) -> EvaluationResult:
        return EvaluationResult(
            evaluator_name=self.name,
            overall_score=0.95,
            passed=True,
            warnings=["Minor tone deviation warning."],
            recommendations=["Keep sentences under 15 words."],
        )


class ExplodingEvaluator(Evaluator):
    """Mock evaluator that raises an unhandled exception during evaluate()."""
    @property
    def name(self) -> str:
        return "ExplodingEvaluator"

    async def evaluate(self, execution_result: ExecutionResult) -> EvaluationResult:
        raise RuntimeError("Catastrophic evaluator engine failure!")


@pytest.mark.smoke
@pytest.mark.functional
def test_load_json_scenario(eval_runner: EvaluationRunner, sample_json_scenario_path: Path) -> None:
    """Test loading and validating a JSON conversation scenario definition."""
    data = eval_runner.load_scenario(sample_json_scenario_path)
    assert data["conversation_id"] == "SCENARIO_FLIGHT_001"
    assert "description" in data
    assert len(data["conversation"]) == 2


@pytest.mark.smoke
@pytest.mark.functional
def test_load_yaml_scenario(eval_runner: EvaluationRunner, sample_yaml_scenario_path: Path) -> None:
    """Test loading and validating a YAML conversation scenario definition."""
    data = eval_runner.load_scenario(sample_yaml_scenario_path)
    assert data["conversation_id"] == "SCENARIO_FLIGHT_002"
    assert len(data["conversation"]) == 2


@pytest.mark.functional
@pytest.mark.negative
def test_load_scenario_missing_file_raises_error(eval_runner: EvaluationRunner) -> None:
    """Test that attempting to load a non-existent scenario file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        eval_runner.load_scenario("test_data/conversations/non_existent.json")


@pytest.mark.functional
@pytest.mark.negative
def test_load_scenario_invalid_json_raises_error(eval_runner: EvaluationRunner, tmp_path: Path) -> None:
    """Test that loading malformed JSON raises a ValueError."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{invalid_json: true,", encoding="utf-8")
    with pytest.raises(ValueError):
        eval_runner.load_scenario(bad_file)


@pytest.mark.asyncio
@pytest.mark.functional
async def test_execute_scenario_no_evaluators(eval_runner: EvaluationRunner, sample_json_scenario_path: Path) -> None:
    """Test executing a voice conversation pipeline without any registered evaluators."""
    scenario = eval_runner.load_scenario(sample_json_scenario_path)
    report = await eval_runner.execute_scenario(scenario)

    assert report.conversation_id == "SCENARIO_FLIGHT_001"
    assert report.execution_result.recognized_text == "Book ticket tomorrow"
    assert "destination" in report.execution_result.assistant_response.lower()
    assert report.execution_result.total_latency > 0.0
    assert len(report.evaluator_results) == 0


@pytest.mark.asyncio
@pytest.mark.functional
async def test_execute_scenario_with_registered_evaluators(
    eval_runner: EvaluationRunner,
    sample_json_scenario_path: Path,
    deepeval_runner: DeepEvalRunner,
) -> None:
    """Test pipeline execution with both DeepEval and a custom enterprise metric evaluator."""
    eval_runner.register_evaluator(deepeval_runner)
    eval_runner.register_evaluator(DummyCustomEvaluator())

    assert len(eval_runner.get_registered_evaluators()) == 2

    scenario = eval_runner.load_scenario(sample_json_scenario_path)
    report = await eval_runner.execute_scenario(scenario)

    assert len(report.evaluator_results) == 2
    assert report.passed is True
    assert report.overall_score > 0.8
    assert "Minor tone deviation warning." in report.warnings[0]


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.negative
async def test_execute_scenario_evaluator_exception_resilience(
    eval_runner: EvaluationRunner,
    sample_json_scenario_path: Path,
    deepeval_runner: DeepEvalRunner,
) -> None:
    """Test that an unhandled exception in one evaluator does not crash the overall runner."""
    eval_runner.register_evaluator(ExplodingEvaluator())
    eval_runner.register_evaluator(deepeval_runner)

    scenario = eval_runner.load_scenario(sample_json_scenario_path)
    report = await eval_runner.execute_scenario(scenario)

    # Both evaluators should be recorded; exploding evaluator recorded with error
    assert len(report.evaluator_results) == 2
    assert report.passed is False
    assert any("Catastrophic evaluator engine failure!" in w for w in report.warnings)
